"""BLIP captions, stored as human-readable metadata next to each photo."""
from __future__ import annotations

import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

MODEL_NAME = "Salesforce/blip-image-captioning-base"
_model = None
_processor = None


def caption(image: Image.Image) -> str:
    global _model, _processor
    if _model is None:
        _processor = BlipProcessor.from_pretrained(MODEL_NAME)
        _model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME).eval()
    inputs = _processor(images=image, return_tensors="pt")
    with torch.no_grad():
        out = _model.generate(**inputs, max_new_tokens=50)
    return _processor.decode(out[0], skip_special_tokens=True).strip()
