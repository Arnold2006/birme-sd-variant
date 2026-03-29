#!/usr/bin/env python3
"""
Flask server for Birme SD Variant.

Serves the static web app and exposes a /api/caption endpoint that runs
images through the JoyCaption Beta One model (fancyfeast/llama-joycaption-beta-one-hf-llava).
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
_compute_dtype = None  # dtype used for pixel_values (set during model load)

MODEL_ID = "fancyfeast/llama-joycaption-beta-one-hf-llava"

CAPTION_PROMPTS = {
    "descriptive": "Write a long detailed description for this image.",
    "training": (
        "Output a stable diffusion prompt that is indistinguishable from a real stable diffusion prompt."
    ),
    "short": "Write a short description for this image.",
}


def _load_model():
    """Load the JoyCaption model in NF4 4-bit quantisation (called inside _model_lock)."""
    global _model, _processor, _model_status, _compute_dtype
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
            _compute_dtype = torch.bfloat16
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=_compute_dtype,
                bnb_4bit_use_double_quant=True,
            )
            _model = LlavaForConditionalGeneration.from_pretrained(
                MODEL_ID,
                quantization_config=quantization_config,
                device_map="auto",
            )
        else:
            # NF4 requires CUDA; fall back to float32 on CPU
            _compute_dtype = torch.float32
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


@app.route("/api/preload", methods=["POST"])
def api_preload():
    """Start loading the JoyCaption model in the background if not already loading."""
    if _model_status == "not_loaded":
        threading.Thread(target=_get_model, daemon=True).start()
    return jsonify({"model_status": _model_status})


@app.route("/api/caption", methods=["POST"])
def api_caption():
    """
    Caption a single image with JoyCaption Beta One.

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
        system_prompt = data.get("system_prompt", "").strip() or "You are a helpful image captioner."

        model, processor = _get_model()

        import torch

        # Build the conversation: system message + plain text user prompt.
        # WARNING: HF's handling of chats on LLaVA models is fragile.
        # Do NOT put {"type": "image"} in the conversation content – pass the
        # image directly to processor() instead to avoid double image tokens.
        conversation = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": prompt_text,
            },
        ]
        convo_string = processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        assert isinstance(convo_string, str)

        # model.device is 'meta' when device_map="auto" is used; derive the
        # real device from the first parameter instead.
        device = next(model.parameters()).device
        # Pass text as a list and image separately – the processor inserts the
        # image tokens automatically.
        inputs = processor(
            text=[convo_string], images=[image], return_tensors="pt"
        ).to(device)

        # Cast pixel_values to the model's compute dtype (bfloat16 on CUDA,
        # float32 on CPU) to match the quantised weights.
        inputs["pixel_values"] = inputs["pixel_values"].to(_compute_dtype)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                suppress_tokens=None,
                use_cache=True,
                temperature=0.6,
                top_p=0.9,
            )

        # Trim the prompt tokens from the front of the output.
        generate_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
        caption = processor.tokenizer.decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
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
