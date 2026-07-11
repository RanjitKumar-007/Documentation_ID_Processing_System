"""Independent raw-text extraction, crash-isolated.

The OCR engine (RapidOCR on ONNX runtime) runs in a SEPARATE Python
subprocess. Native-library crashes (segfaults), hangs, or import
errors in the OCR stack therefore can NEVER take down the main app:
any child failure simply yields '' and the pipeline continues
without the cross-check layer.

Set OCR_ENABLED = False to skip OCR entirely.
"""

import json
import subprocess
import sys

OCR_ENABLED = True
TIMEOUT_S = 120
_CHILD_SCRIPT = r"""
import json, sys

def texts_from(result):
    if result is None:
        return []
    txts = getattr(result, "txts", None)
    if txts is not None:
        return [t for t in txts if t]
    if isinstance(result, tuple):
        result = result[0]
    if not result:
        return []
    out = []
    for entry in result:
        try:
            out.append(entry[1])
        except (TypeError, IndexError):
            continue
    return [t for t in out if isinstance(t, str)]

try:
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError:
        from rapidocr import RapidOCR
    engine = RapidOCR()
    all_text = ""
    for path in sys.argv[1:]:
        try:
            for t in texts_from(engine(path)):
                all_text += t + " "
        except Exception:
            continue
    print(json.dumps({"text": all_text}))
except Exception:
    print(json.dumps({"text": ""}))
"""


def extract_with_ocr(image_paths):
    """Return concatenated raw text from all pages ('' on any failure)."""
    if not OCR_ENABLED or not image_paths:
        return ""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _CHILD_SCRIPT, *image_paths],
            capture_output=True, text=True, timeout=TIMEOUT_S,
        )
        for line in reversed(proc.stdout.strip().splitlines()):
            line = line.strip()
            if line.startswith("{"):
                return json.loads(line).get("text", "")
        return ""
    except Exception:
        return ""
