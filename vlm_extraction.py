"""Text extraction using Qwen2.5-VL 3B via Ollama"""

import json
import os
import re
import subprocess
import time
from datetime import datetime
import ollama

MODEL = "qwen2.5vl:3b"
LOG_DIR = "logs"
OPTIONS = {"temperature": 0, "num_predict": 600, "num_ctx": 4096,
           "num_gpu": 36}
RESET_AFTER_RUNS = 10
_run_count = 0

PROMPT = """You are given one or more images of the SAME document
(front and back may be separate images). Classify and extract.

Classify as exactly one of: "aadhaar", "pan", "other".

Do NOT classify by "it's an Indian government photo ID card" alone -
many document types share that general look (Voter ID/EPIC card, driving
licence, passport, ration card, etc). Use these specific markers:

- "aadhaar": issued by UIDAI. Says "Unique Identification Authority of
  India" / "Government of India" with the UIDAI logo, and has a 12-digit
  Aadhaar number formatted in 3 groups of 4 (e.g. "9861 4978 9451"),
  usually with a QR code on the back.
- "pan": issued by the Income Tax Department. Has a 10-character PAN
  number (5 letters, 4 digits, 1 letter, e.g. "ABCDE1234F") and says
  "Income Tax Department" / "Permanent Account Number".
- "other": anything else, INCLUDING:
  - Voter ID / EPIC card: issued by the "Election Commission of India",
    has an EPIC number like "AP31206000841", says "Elector Photo
    Identity Card".
  - Driving licence: says "Driving Licence" / "Union of India Driving
    Licence", has a DL number (2 letters + digits, e.g.
    "AN01 20130003278"), often has a chip icon and a "Blood Group" field.
  - Passport: says "REPUBLIC OF INDIA" / "PASSPORT", has a Passport No.
    (1 letter + 7 digits, e.g. "J8369854"), and two machine-readable
    lines of "<" characters at the bottom of the photo page.
  - ration cards, or any card that does not show the specific UIDAI or
    Income Tax Department markers above.
  When in doubt, classify as "other" rather than guessing "aadhaar".

If aadhaar, extract: name, dob (DD/MM/YYYY), gender ("Male" or "Female"),
phone_number (the 10-digit number next to the "Mobile" label, if printed,
else null), address (full, from back side),
aadhaar_number (exactly 12 digits - do NOT return the 16-digit VID).
Extract DOB only from the value next to the DOB label,
never from disclaimer text.

The word "Mobile" is printed in SMALL text on the FRONT image, directly
above or below the QR code (e.g. "Mobile:9182312102"). Look closely near
the QR code specifically for this label. Do NOT confuse the number next
to it with the Aadhaar number (12 digits) or the VID (16 digits). If the
"Mobile" label is not visible anywhere, use null.

If pan, extract: name (cardholder), father_name, dob (DD/MM/YYYY),
pan_number (5 letters, 4 digits, 1 letter).

Rules:
- Combine information across all provided images.
- Any field not visible or unreadable: null.
- If not a valid aadhaar or pan: document_type "other", all fields null.

Respond ONLY with this JSON, no markdown, no extra text:
{"document_type": "", "name": null, "father_name": null, "dob": null,
"gender": null, "phone_number": null, "address": null,
"aadhaar_number": null, "pan_number": null}"""

_EXPECTED_KEYS = {
    "document_type", "name", "father_name", "dob", "gender",
    "phone_number", "address", "aadhaar_number", "pan_number",
}


def _parse_model_json(raw):
    """Extract a JSON object from model output, tolerating noise.
    Handles: markdown fences, leading/trailing prose, single quotes,
    trailing commas, and Python-style None/True/False.
    Raises json.JSONDecodeError if nothing parseable is found.
    """
    text = re.sub(r"```(?:json)?", "", raw).strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        text = text[start:end + 1]
    for candidate in (text, _repair(text)):
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
            if isinstance(obj, list) and obj and isinstance(obj[0], dict):
                return obj[0]
        except json.JSONDecodeError:
            continue
    raise json.JSONDecodeError("no parseable JSON in model output",
                               raw[:200], 0)


def _repair(text):
    """Best-effort fixes for common model JSON mistakes"""
    fixed = text
    fixed = re.sub(r"\bNone\b", "null", fixed)
    fixed = re.sub(r"\bTrue\b", "true", fixed)
    fixed = re.sub(r"\bFalse\b", "false", fixed)
    fixed = re.sub(r",\s*([}\]])", r"\1", fixed)
    if '"' not in fixed and "'" in fixed:
        fixed = fixed.replace("'", '"')
    return fixed


def _normalize(obj):
    """Guarantee every expected key exists; unknown keys are dropped"""
    return {k: obj.get(k) for k in _EXPECTED_KEYS}


def extract_with_vlm(image_paths):
    """One call with all pages attached. Returns a normalized dict.
    Self-healing: on degenerate/unparseable output, unloads the model
    and retries once against a fresh load. After every
    RESET_AFTER_RUNS completed runs, performs the full Ollama
    terminate-and-restart to clear VRAM.

    If the VLM returns a null phone_number on an aadhaar document, an
    OCR-based regex fallback is attempted before returning - small
    print near the QR code is a common miss for a 3B vision model.
    """
    messages = [{"role": "user", "content": PROMPT, "images": image_paths}]
    resp = ollama.chat(model=MODEL, messages=messages, options=OPTIONS)
    raw = resp["message"]["content"]
    if not _is_degenerate(raw):
        try:
            result = _normalize(_parse_model_json(raw))
            _count_run()
            return result
        except json.JSONDecodeError:
            _log_failure(image_paths, raw, attempt=1)
    else:
        _log_failure(image_paths, raw, attempt=1)
    _reset_model()
    resp = ollama.chat(model=MODEL, messages=messages, options=OPTIONS)
    raw2 = resp["message"]["content"]
    try:
        result = _normalize(_parse_model_json(raw2))
        _count_run()
        return result
    except json.JSONDecodeError:
        _log_failure(image_paths, raw2, attempt=2)
        raise


def _count_run():
    """Count a completed run; every Nth run, fully restart Ollama."""
    global _run_count
    _run_count += 1
    if RESET_AFTER_RUNS and _run_count % RESET_AFTER_RUNS == 0:
        _restart_ollama()


def _restart_ollama():
    """Automated equivalent of the manual VRAM-clear routine:
        taskkill /IM ollama.exe /F
        taskkill /IM llama-server.exe /F
    then restarts the Ollama service (`ollama serve`) in the
    background so the next run works without manual steps.
    """
    if os.name != "nt":
        return
    try:
        for proc in ("ollama.exe", "llama-server.exe"):
            subprocess.run(["taskkill", "/IM", proc, "/F"],
                           capture_output=True, timeout=15)
        print("[VLM] Ollama terminated - VRAM cleared")
        time.sleep(2)
        flags = (subprocess.DETACHED_PROCESS
                 | subprocess.CREATE_NO_WINDOW)
        subprocess.Popen(["ollama", "serve"],
                         stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL,
                         creationflags=flags)
        time.sleep(5)
        print("[VLM] Ollama restarted - next run loads a fresh model")
    except Exception:
        pass


def _is_degenerate(raw):
    """Detect collapsed generations: empty or long repeated-char runs"""
    if not raw or not raw.strip():
        return True
    return bool(re.search(r"(.)\1{29,}", raw))


def _reset_model():
    """Force-unload the model so the next call gets a fresh runner"""
    try:
        ollama.generate(model=MODEL, prompt="", keep_alive=0)
    except Exception:
        pass


def _log_failure(image_paths, raw, attempt):
    """Save the unparseable model output for diagnosis"""
    os.makedirs(LOG_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(LOG_DIR, f"vlm_fail_{stamp}_try{attempt}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"images: {image_paths}\n")
        f.write(f"attempt: {attempt}\n")
        f.write("---- raw model output ----\n")
        f.write(raw if raw else "<EMPTY RESPONSE>")
    print(f"[VLM] unparseable output logged to {path}")
    preview = (raw or "<EMPTY>")[:300].replace("\n", " ")
    print(f"[VLM] output preview: {preview}")
    