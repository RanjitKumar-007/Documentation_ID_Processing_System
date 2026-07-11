# 🪪 Intelligent Document ID Processing System

An offline AI system that reads **Aadhaar cards and PAN cards** from
images or PDFs, classifies them, extracts the person's details and
photo, and returns everything as structured JSON — with a per-field
confidence score.

Everything runs **locally on your machine**. No document data ever
leaves your computer, which matters because Aadhaar/PAN are sensitive
identity documents.

---

## What it does

1. You upload a document (JPG / JPEG / PNG / PDF — front and back
   pages of a PDF are both read).
2. The system classifies it as **Aadhaar**, **PAN**, or **Invalid
   input** (any other document, e.g. voter ID, driving licence, passport, school ID, etc.).
3. If valid, it extracts:

   | Aadhaar | PAN |
   |---|---|
   | Name | Name |
   | Date of Birth | Father's Name |
   | Gender | Date of Birth |
   | Phone Number (if printed) | PAN Number |
   | Address | |
   | Aadhaar Number | |

4. It also crops out the **person's photo** from the card.
5. Each field gets a **confidence score** (color-coded in the UI:
   green ≥ 90%, amber 80–89%, red < 80% = "verify manually").
6. The result is saved as a JSON file in the `output/` folder.

---

## How it works (architecture)

```
Upload (image / PDF)
      |
      v
Input handler ......... PDF pages -> images, smart resize
      |
      v
Preprocessing ......... auto-fixes brightness, contrast, noise,
      |                 blur, skew; rejects unreadably blurred images
      v
Qwen2.5-VL (Ollama) ... vision-language model classifies the document
      |                 and extracts all text fields (runs on GPU)
      v
RapidOCR .............. independent second reading of the text
      |                 (CPU, crash-isolated subprocess)
      v
Validation layer ...... regex format checks, Verhoeff checksum on the
      |                 Aadhaar number, OCR cross-check, per-field
      |                 confidence scoring, date normalization
      v
output/<name>_<timestamp>.json  +  photos/<photo crop>
      |
      v
Gradio UI ............. document type badge, field table, photo
```

Key idea: the VLM understands *which* text is which field; the OCR
engine provides an independent character-level reading. When both
agree on a value, confidence is high; when they disagree, the field
is flagged for manual review. Wrong identity data silently accepted
is the worst outcome — this design makes errors *visible*.

---

## Requirements

- Windows 10/11, Python 3.10 (conda recommended)
- NVIDIA GPU with 6 GB+ VRAM (tested on RTX 3050 6 GB)
- ~8 GB free disk (model + packages)
- Internet **once**, for setup only

## Installation (one-time)

```bash
# 1. Install Ollama from https://ollama.com, then pull the model:
ollama pull qwen2.5vl:3b

# 2. Create and activate an environment:
conda create -n doc_classifier python=3.10 -y
conda activate doc_classifier

# 3. Install the Python packages:
pip install -r requirements.txt
```

`requirements.txt`:

```
numpy
gradio
ollama
pillow
opencv-python
pymupdf
rapidocr-onnxruntime
```

## Running

```bash
conda activate doc_classifier
cd <project folder>
python main.py
```

Open **http://127.0.0.1:7860** in your browser:

1. **Browse** an image or PDF
2. Click **▶ Run** (button shows "⏳ Processing..." while working)
3. Read the results: document type, extracted fields with confidence,
   and the cropped photo
4. The JSON is saved automatically to `output/`

First run after startup is slower (~20 s) while the model loads.
A typical document takes roughly 10–30 seconds.

## Output format

One JSON file per processed document, e.g.
`output/aadhaar_sample_20260710_101530.json`:

```json
{
  "document_type": "Aadhaar",
  "fields": {
    "name": "Anjali",
    "dob": "1999-09-18",
    "gender": "Female",
    "phone_number": "9582539507",
    "address": "D/O Mukesh Kumar, ... Delhi - 110093",
    "aadhaar_number": "226816223671"
  },
  "confidence": {
    "name": 0.95,
    "dob": 0.97,
    "gender": 0.95,
    "phone_number": 0.85,
    "address": 0.88,
    "aadhaar_number": 0.99
  },
  "photo_path": "photos/....jpg"
}
```

Fields that are not printed or not readable are simply omitted.
Invalid documents produce `{"document_type": "Unknown", ...}`.

## Project structure

```
main.py             entry point — run only this
ui.py               Gradio interface (dark theme, confidence badges)
utils.py            input handling, validation, checksum, storage
preprocessing.py    adaptive image quality repair
vlm_extraction.py   Qwen2.5-VL prompt + call, self-healing recovery
ocr_extraction.py   RapidOCR in a crash-isolated subprocess
face_detection.py   photo crop (OpenCV)
output/             one JSON per processed document (auto-created)
photos/             cropped card photos (auto-created)
logs/               raw model output on failures (auto-created)
```

## Configuration knobs

| Setting | File | Default | Meaning |
|---|---|---|---|
| `num_gpu` | vlm_extraction.py | 36 | model layers on GPU; lower it (29 / 20) if you see garbage output on low-VRAM machines |
| `RESET_AFTER_RUNS` | vlm_extraction.py | 10 | auto-restart Ollama every N documents to clear VRAM (0 = off) |
| `OCR_ENABLED` | ocr_extraction.py | True | set False to skip the OCR cross-check entirely |
| `BLUR_REJECT` etc. | preprocessing.py | — | image-quality thresholds |

## Troubleshooting

- **"Could not read document"** — click Run again (the system
  auto-recovers by reloading the model). Raw model output is saved
  in `logs/` for diagnosis.
- **Garbage output on several runs in a row** — VRAM pressure. Close
  GPU-heavy apps (browser tabs!), or lower `num_gpu` to 29 or 20.
- **VRAM stuck after a crash** — `taskkill /IM llama-server.exe /F`;
  the service reloads the model on the next run.
- **All confidences stuck at 80–85%** — the OCR layer isn't running;
  check `pip show rapidocr-onnxruntime`.

## Privacy & limitations

- Fully offline after setup: UI, model, OCR, and storage are all
  local. No network calls at runtime.
- Aadhaar/PAN are sensitive personal data — in production, stored
  results would need encryption and masked display per UIDAI
  guidelines; this prototype stores plain JSON for evaluation.
- Occluded or glare-covered text cannot be recovered and is returned
  as missing rather than guessed — by design, for identity data.
- The 3B quantized model can misread characters on low-quality
  images; the checksum + OCR cross-check turns most such errors into
  visible low-confidence flags instead of silent mistakes.
