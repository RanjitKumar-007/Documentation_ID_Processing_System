"""Gradio interface.

Layout: left = upload card + details card; right = document type +
photo cards. Dark theme with interactive styling, friendly field
labels, and color-coded confidence badges.
"""

import json
import gradio as gr
from utils import prepare_inputs, validate, save_result
from preprocessing import preprocess
from vlm_extraction import extract_with_vlm
from ocr_extraction import extract_with_ocr
from face_detection import extract_photo

VERSION = "v1"

FIELD_LABELS = {
    "name": "Name",
    "father_name": "Father's Name",
    "dob": "Date of Birth",
    "gender": "Gender",
    "phone_number": "Phone Number",
    "address": "Address",
    "aadhaar_number": "Aadhaar Number",
    "pan_number": "PAN Number",
}

DARK_CSS = """
.gradio-container, body {
    background: #0f1117 !important;
    color: #e5e7eb !important;
    font-family: 'Segoe UI', Inter, Roboto, 'Helvetica Neue',
                 Arial, sans-serif !important;
}
.gradio-container * {
    font-family: inherit !important;
    --body-background-fill: #0f1117;
    --background-fill-primary: #171923;
    --background-fill-secondary: #1f2330;
    --block-background-fill: #171923;
    --input-background-fill: #1f2330;
    --body-text-color: #e5e7eb;
    --block-label-text-color: #9ca3af;
    --block-title-text-color: #e5e7eb;
    --border-color-primary: #2d3348;
    --block-border-color: #2d3348;
    --input-border-color: #2d3348;
    --color-accent: #f97316;
    --link-text-color: #fb923c;
    --button-secondary-background-fill: #1f2330;
    --button-secondary-text-color: #e5e7eb;
    --button-secondary-border-color: #2d3348;
}
.gradio-container h1, .gradio-container h2,
.gradio-container h3, .gradio-container p,
.gradio-container span, .gradio-container label {
    color: #e5e7eb !important;
}
/* Floating block-label chips */
.gradio-container [data-testid="block-label"],
.gradio-container [data-testid="block-label"] *,
.gradio-container .label-wrap,
.gradio-container .block > .label,
.gradio-container .block > label,
.gradio-container .block > label > span {
    background: #1f2330 !important;
    color: #9ca3af !important;
    border-color: #2d3348 !important;
}
/* File-upload internals */
.gradio-container [data-testid="file"],
.gradio-container [data-testid="file"] *,
.gradio-container .file-preview,
.gradio-container .file-preview *,
.gradio-container input[type="file"],
.gradio-container .upload-container,
.gradio-container .upload-container * {
    background: #1f2330 !important;
    color: #e5e7eb !important;
    border-color: #2d3348 !important;
}
/* Image component: dark frame everywhere, no white edges */
.gradio-container [data-testid="image"],
.gradio-container [data-testid="image"] > div,
.gradio-container [data-testid="image"] .image-frame,
.gradio-container .image-container,
.gradio-container .image-container > div,
.gradio-container .image-frame,
.gradio-container .image-frame img,
.gradio-container .empty {
    background: #171923 !important;
    border-color: #2d3348 !important;
    color: #6b7280 !important;
}
/* Cards */
.result-group {
    background: #171923 !important;
    border: 1px solid #2d3348 !important;
    border-radius: 12px !important;
    padding: 16px !important;
    margin: 0 0 16px 0 !important;
    transition: border-color .2s ease, box-shadow .2s ease,
                transform .15s ease;
}
.result-group:hover {
    border-color: #f97316 !important;
    box-shadow: 0 4px 18px rgba(249, 115, 22, .12);
    transform: translateY(-1px);
}
/* Primary button feedback */
.gradio-container button.primary,
.gradio-container .primary {
    transition: filter .15s ease, transform .1s ease,
                box-shadow .2s ease !important;
}
.gradio-container button.primary:hover {
    filter: brightness(1.12);
    box-shadow: 0 4px 16px rgba(249, 115, 22, .35);
}
.gradio-container button.primary:active { transform: scale(.985); }
#col-left, #col-right { gap: 0 !important; align-self: flex-start; }

/* ---- Friendly result rendering ---- */
.doc-badge {
    display: inline-flex; align-items: center; gap: 8px;
    padding: 8px 18px; border-radius: 999px;
    font-size: 1.05rem; font-weight: 600;
}
.doc-ok    { background: rgba(34,197,94,.15);  color: #4ade80 !important;
             border: 1px solid rgba(34,197,94,.4); }
.doc-bad   { background: rgba(239,68,68,.15);  color: #f87171 !important;
             border: 1px solid rgba(239,68,68,.4); }
.doc-warn  { background: rgba(245,158,11,.15); color: #fbbf24 !important;
             border: 1px solid rgba(245,158,11,.4); }
.field-row {
    display: flex; align-items: baseline; gap: 12px;
    padding: 9px 4px; border-bottom: 1px solid #232838;
}
.field-row:last-child { border-bottom: none; }
.field-label {
    flex: 0 0 140px; color: #9ca3af !important; font-size: .88rem;
}
.field-value { flex: 1; color: #e5e7eb !important; font-weight: 500; }
.conf-pill {
    flex: 0 0 auto; padding: 2px 10px; border-radius: 999px;
    font-size: .75rem; font-weight: 600;
}
.conf-high { background: rgba(34,197,94,.15);  color: #4ade80 !important; }
.conf-mid  { background: rgba(245,158,11,.15); color: #fbbf24 !important; }
.conf-low  { background: rgba(239,68,68,.15);  color: #f87171 !important; }
.hint-text { color: #6b7280 !important; font-size: .9rem; }
"""

FORCE_DARK_JS = """
() => {
    const url = new URL(window.location);
    if (url.searchParams.get('__theme') !== 'dark') {
        url.searchParams.set('__theme', 'dark');
        window.location.href = url.href;
    }
}
"""


def _conf_pill(c):
    cls = "conf-high" if c >= 0.9 else "conf-mid" if c >= 0.8 else "conf-low"
    return f'<span class="conf-pill {cls}">{int(round(c * 100))}%</span>'


def _render_details(result):
    rows = []
    for k, v in result["fields"].items():
        c = result["confidence"].get(k, 0)
        label = FIELD_LABELS.get(k, k)
        rows.append(
            f'<div class="field-row">'
            f'<span class="field-label">{label}</span>'
            f'<span class="field-value">{v}</span>'
            f'{_conf_pill(c)}</div>'
        )
    if not rows:
        return '<p class="hint-text">No fields could be extracted.</p>'
    low = [FIELD_LABELS.get(k, k)
           for k, v in result["confidence"].items() if v < 0.8]
    html = "".join(rows)
    if low:
        html += (f'<p class="hint-text" style="margin-top:10px">⚠️ Please '
                 f'verify manually: {", ".join(low)}</p>')
    return html


def _badge(text, kind):
    return f'<span class="doc-badge doc-{kind}">{text}</span>'


def process(file):
    """Run the full pipeline for one uploaded file."""
    if file is None:
        return (_badge("⚠ No file selected", "warn"),
                '<p class="hint-text">Browse a document, then '
                'click Run.</p>', None)
    paths = prepare_inputs(file.name)
    for p in paths:
        report = preprocess(p)
        if report["rejected"]:
            return (_badge("✖ Image too blurred", "bad"),
                    '<p class="hint-text">Please re-upload a clearer '
                    'image.</p>', None)
    try:
        raw = extract_with_vlm(paths)
    except (json.JSONDecodeError, KeyError):
        return (_badge("⚠ Could not read document", "warn"),
                '<p class="hint-text">The model could not produce a '
                'readable result. Please click Run again.</p>', None)
    ocr_text = extract_with_ocr(paths)
    result = validate(raw, ocr_text)
    result["source_file"] = paths[0] if paths else None
    result["photo_path"] = (
        extract_photo(paths)
        if result["document_type"] != "Unknown" else None
    )
    save_result(result, file.name)
    dt = result["document_type"]
    if dt == "Unknown":
        return (_badge("✖ Invalid input", "bad"),
                '<p class="hint-text">This does not appear to be an '
                'Aadhaar or PAN card.</p>', None)
    return (_badge(f"✔ {dt}", "ok"),
            _render_details(result),
            result["photo_path"])


def _btn_running():
    return gr.Button("⏳ Processing... please wait", interactive=False)


def _btn_ready():
    return gr.Button("▶ Run", variant="primary", interactive=True)


def build_app():
    with gr.Blocks(title="Intelligent Document ID Processing System",
                   analytics_enabled=False) as app:
        gr.Markdown("# 🪪 Intelligent Document ID Processing System")
        with gr.Row(equal_height=False):
            # ---------- Left: upload + details ----------
            with gr.Column(scale=1, min_width=420, elem_id="col-left"):
                with gr.Group(elem_classes="result-group"):
                    gr.Markdown("### 📤 Upload")
                    file_in = gr.File(
                        label="Browse image or PDF",
                        file_types=[".jpg", ".jpeg", ".png", ".pdf"],
                    )
                    run_btn = gr.Button("▶ Run", variant="primary",
                                        size="lg")
                with gr.Group(elem_classes="result-group"):
                    gr.Markdown("### 📋 Details")
                    details_out = gr.HTML(
                        '<p class="hint-text">Extracted fields will '
                        'appear here.</p>')
            # ---------- Right: document type + photo ----------
            with gr.Column(scale=1, min_width=420, elem_id="col-right"):
                with gr.Group(elem_classes="result-group"):
                    gr.Markdown("### 📄 Document type")
                    doc_type_out = gr.HTML(
                        '<p class="hint-text">Upload a document and '
                        'click Run.</p>')
                with gr.Group(elem_classes="result-group"):
                    gr.Markdown("### 🖼️ Photo")
                    photo_out = gr.Image(
                        show_label=False, height=260, interactive=False,
                    )
        run_btn.click(
            _btn_running, inputs=None, outputs=run_btn, queue=False,
        ).then(
            process, inputs=file_in,
            outputs=[doc_type_out, details_out, photo_out],
        ).then(
            _btn_ready, inputs=None, outputs=run_btn, queue=False,
        )
    # Dark theme applies regardless of how main.py launches the app
    _orig_launch = app.launch
    def _launch(*args, **kwargs):
        kwargs.setdefault("css", DARK_CSS)
        kwargs.setdefault("js", FORCE_DARK_JS)
        try:
            return _orig_launch(*args, **kwargs)
        except TypeError:
            kwargs.pop("css", None)
            kwargs.pop("js", None)
            return _orig_launch(*args, **kwargs)
    app.launch = _launch
    return app
