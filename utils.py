"""Input handling, validation, and storage"""

import os
import re
import json
from datetime import datetime
import fitz
from PIL import Image

OUTPUT_DIR = "output"    # JSON file per processed file (name + timestamp)
PHOTO_DIR = "photos"
for _d in (OUTPUT_DIR, PHOTO_DIR):
    os.makedirs(_d, exist_ok=True)

SCHEMA = {
    "aadhaar": ["name", "dob", "gender", "phone_number", "address",
                "aadhaar_number"],
    "pan": ["name", "father_name", "dob", "pan_number"],
}


# ---------------- Input handling ----------------

def pdf_to_images(path):
    """Convert every page of a PDF to a JPG image (front + back sides)"""
    doc = fitz.open(path)
    paths = []
    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=200)
        p = f"{path[:-4]}_page{i + 1}.jpg"
        pix.save(p)
        paths.append(p)
    return paths


def resize_image(path, max_side=1280, min_side=600):
    """Shrink large images; Lanczos-upscale tiny ones for better OCR"""
    img = Image.open(path)
    if max(img.size) < min_side:
        s = min_side / max(img.size)
        img = img.resize(
            (int(img.width * s), int(img.height * s)), Image.LANCZOS)
    else:
        img.thumbnail((max_side, max_side))
    img.convert("RGB").save(path, quality=92)
    return path


def prepare_inputs(file_path):
    """jpg/png -> [one image]; pdf -> [image per page]"""
    if file_path.lower().endswith(".pdf"):
        return [resize_image(p, max_side=1600)
                for p in pdf_to_images(file_path)]
    return [resize_image(file_path)]


# ---------------- Verhoeff checksum (Aadhaar) ----------------

_d = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
      [2, 3, 4, 0, 1, 7, 8, 9, 5, 6], [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
      [4, 0, 1, 2, 3, 9, 5, 6, 7, 8], [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
      [6, 5, 9, 8, 7, 1, 0, 4, 3, 2], [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
      [8, 7, 6, 5, 9, 3, 2, 1, 0, 4], [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]]
_p = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
      [5, 8, 0, 3, 7, 9, 6, 1, 4, 2], [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
      [9, 4, 5, 3, 1, 2, 6, 8, 7, 0], [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
      [2, 7, 9, 3, 8, 0, 6, 4, 1, 5], [7, 0, 4, 6, 9, 1, 3, 2, 5, 8]]


def verhoeff_valid(num: str) -> bool:
    c = 0
    for i, digit in enumerate(reversed(num)):
        c = _d[c][_p[i % 8][int(digit)]]
    return c == 0


# ---------------- Validation layer ----------------

def normalize_date(dob):
    """Normalize DD/MM/YYYY to ISO format YYYY-MM-DD; None if invalid"""
    m = re.match(r"^(\d{2})/(\d{2})/(\d{4})$", str(dob or ""))
    if not m:
        return None
    day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if not (1 <= month <= 12 and 1 <= day <= 31 and 1900 <= year <= 2026):
        return None
    return f"{year:04d}-{month:02d}-{day:02d}"


def _to_iso_date(dob):
    """Normalize DD/MM/YYYY -> YYYY-MM-DD (assignment output format)"""
    m = re.match(r"^(\d{2})/(\d{2})/(\d{4})$", str(dob or ""))
    if not m:
        return None
    day, month, year = m.groups()
    if not (1 <= int(day) <= 31 and 1 <= int(month) <= 12
            and 1900 <= int(year) <= 2100):
        return None
    return f"{year}-{month}-{day}"


def validate(r, ocr_text=""):
    """Format checks, checksum, OCR cross-check, per-field confidence.
    Returns the assignment output format:
      {"document_type": "Aadhaar"|"PAN"|"Unknown",
       "fields": {...non-null fields...},
       "confidence": {...same keys, numeric 0-1...}}
    """
    dt = r.get("document_type")
    if dt not in SCHEMA:
        return {"document_type": "Unknown", "fields": {}, "confidence": {}}
    raw_digits = re.sub(r"\D", "", ocr_text)
    ocr_upper = ocr_text.upper().replace(" ", "")
    fields, conf = {}, {}
    dob_ddmm = str(r.get("dob") or "")
    dob_iso = _to_iso_date(dob_ddmm)
    if dob_iso:
        c = 0.85
        if raw_digits and re.sub(r"\D", "", dob_ddmm) in raw_digits:
            c = 0.97
        fields["dob"] = dob_iso
        conf["dob"] = c
    for key in ("name", "father_name"):
        if key == "father_name" and dt != "pan":
            continue
        val = str(r.get(key) or "").strip()
        if val and re.match(r"^[A-Za-z][A-Za-z .']{1,60}$", val):
            c = 0.80
            if ocr_text:
                if val.upper().replace(" ", "") in ocr_upper:
                    c = 0.95
                else:
                    c = 0.65
            fields[key] = val
            conf[key] = c

    if dt == "aadhaar":
        # Aadhaar number: 12 digits + Verhoeff + OCR cross-check
        num = re.sub(r"\D", "", str(r.get("aadhaar_number") or ""))
        if len(num) != 12 and ocr_text:
            m = re.search(r"\b\d{4}\s?\d{4}\s?\d{4}\b", ocr_text)
            if m:
                num = re.sub(r"\D", "", m.group())
        if len(num) == 12:
            c = 0.60
            if verhoeff_valid(num):
                c = 0.85
                if raw_digits and num in raw_digits:
                    c = 0.99
            fields["aadhaar_number"] = num
            conf["aadhaar_number"] = c
        g = str(r.get("gender") or "").lower()
        gender = ("Male" if g.startswith("m")
                  else "Female" if g.startswith("f") else None)
        if gender:
            fields["gender"] = gender
            conf["gender"] = 0.95
        ph = re.sub(r"\D", "", str(r.get("phone_number") or ""))
        aadhaar_num = fields.get("aadhaar_number", "")
        if (len(ph) == 10 and ph[0] in "6789"
                and (not aadhaar_num or ph not in aadhaar_num)):
            fields["phone_number"] = ph
            conf["phone_number"] = 0.85
        addr = str(r.get("address") or "").strip()
        if addr:
            c = 0.75
            if re.search(r"\b\d{6}\b", addr):
                c = 0.88
            fields["address"] = addr
            conf["address"] = c

    else:
        # PAN number: format regex + OCR cross-check
        pan = str(r.get("pan_number") or "").upper().replace(" ", "")
        if not re.match(r"^[A-Z]{5}\d{4}[A-Z]$", pan) and ocr_text:
            m = re.search(r"\b[A-Z]{5}\d{4}[A-Z]\b", ocr_text.upper())
            if m:
                pan = m.group()
        if re.match(r"^[A-Z]{5}\d{4}[A-Z]$", pan):
            c = 0.80
            if ocr_upper and pan in ocr_upper:
                c = 0.99
            fields["pan_number"] = pan
            conf["pan_number"] = c
    label = "Aadhaar" if dt == "aadhaar" else "PAN"
    return {"document_type": label, "fields": fields, "confidence": conf}


# ---------------- Storage ----------------

def save_result(result, source_file):
    """Write one JSON per processed file into output/.
    Filename: <original name>_<YYYYMMDD_HHMMSS>.json
    Returns the written path.
    """
    base = os.path.splitext(os.path.basename(source_file))[0]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(OUTPUT_DIR, f"{base}_{stamp}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return out_path
