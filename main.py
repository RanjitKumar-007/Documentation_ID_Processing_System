"""Entry point. Run only this file:  python main.py

Pipeline (fully offline):
  Gradio UI -> input handler + preprocessing
            -> Qwen2.5-VL extraction (Ollama, GPU)
            -> RapidOCR raw text (CPU, crash-isolated subprocess)
            -> face photo crop (OpenCV)
            -> validation + per-field confidence
            -> output/<name>_<timestamp>.json -> Gradio result panels

Prerequisites (one-time, needs internet):
  pip install gradio ollama pillow opencv-python pymupdf numpy
  pip install rapidocr-onnxruntime
  ollama pull qwen2.5vl:3b
"""

from ui import build_app, VERSION

if __name__ == "__main__":
    print(f"Intelligent Document ID Processing System {VERSION}")
    app = build_app()
    app.launch()
