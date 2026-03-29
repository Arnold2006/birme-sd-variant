#!/usr/bin/env python3
"""
Flask server for Birme SD Variant.

Serves the static web app and exposes a /api/caption endpoint that runs
images through the JoyCaption Alpha Two model (fancyfeast/joy-caption-alpha-two).
"""

import base64
import io
import os
import threading

from flask import Flask, jsonify, request, send_from_directory
from PIL import Image

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__, static_folder=".")

# ---------------------------------------------------------------------------
# JoyCaption model – loaded lazily on the first captioning request so that
# the server starts up fast and the model is only downloaded when needed.
# ---------------------------------------------------------------------------
_model = None
_processor = None
_model_lock = threading.Lock()
_model_status = "not_loaded"  # not_loaded | loading | ready | error:<msg>

MODEL_ID = "fancyfeast/llama-joycaption-beta-one-hf-llava"

CAPTION_PROMPTS = {
    "descriptive": "Write a descriptive caption for this image in a formal tone.",
    "training": (
        "Write a stable diffusion training caption for this image. "
        "Focus on visual details such as subject, style, colours, lighting and composition."
    ),
    "short": "Write a short, concise caption for this image in one sentence.",
}


def _load_model():
    """Load the JoyCaption model in NF4 4-bit quantisation (called inside _model_lock)."""
    global _model, _processor, _model_status
    _model_status = "loading"
    try:
        import torch
        from transformers import (
            AutoProcessor,
            BitsAndBytesConfig,
            LlavaForConditionalGeneration,
        )

        _processor = AutoProcessor.from_pretrained(MODEL_ID)

        if torch.cuda.is_available():
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            _model = LlavaForConditionalGeneration.from_pretrained(
                MODEL_ID,
                quantization_config=quantization_config,
                device_map="auto",
            )
        else:
            # NF4 requires CUDA; fall back to float32 on CPU
            _model = LlavaForConditionalGeneration.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float32,
            ).to("cpu")

        _model.eval()
        _model_status = "ready"
    except Exception as exc:
        _model_status = f"error: {exc}"
        raise


def _get_model():
    """Return the loaded model, loading it first if necessary."""
    global _model, _processor
    with _model_lock:
        if _model is None:
            _load_model()
    return _model, _processor


# ---------------------------------------------------------------------------
# Routes – static files
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(".", filename)


# ---------------------------------------------------------------------------
# Routes – JoyCaption API
# ---------------------------------------------------------------------------
@app.route("/api/status")
def api_status():
    """Return the current model loading status."""
    return jsonify({"pinokio": True, "model_status": _model_status})


@app.route("/api/caption", methods=["POST"])
def api_caption():
    """
    Caption a single image with JoyCaption Alpha Two.

    Request JSON:
        {
          "image": "<data-URL or raw base64>",
          "caption_type": "descriptive" | "training" | "short"   (optional)
        }

    Response JSON:
        { "caption": "<text>" }
    or on error:
        { "error": "<message>" }
    """
    try:
        data = request.get_json(force=True)
        if not data or "image" not in data:
            return jsonify({"error": "No image provided"}), 400

        # Accept "data:image/jpeg;base64,..." or raw base64
        raw = data["image"]
        if "," in raw:
            raw = raw.split(",", 1)[1]
        img_bytes = base64.b64decode(raw)
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        caption_type = data.get("caption_type", "descriptive")
        prompt_text = CAPTION_PROMPTS.get(caption_type, CAPTION_PROMPTS["descriptive"])

        model, processor = _get_model()

        import torch

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(
            model.device
        )

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
            )

        caption = processor.decode(
            output_ids[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        ).strip()

        return jsonify({"caption": caption})

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7861))
    print(f"Starting Birme SD Variant server on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
