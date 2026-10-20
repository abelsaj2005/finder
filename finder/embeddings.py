"""CLIP image/text embeddings (ViT-B/32, 512-d, L2-normalized)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

MODEL_NAME = "openai/clip-vit-base-patch32"
_model = None
_processor = None


def _load():
    global _model, _processor
    if _model is None:
        _model = CLIPModel.from_pretrained(MODEL_NAME).eval()
        _processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    return _model, _processor


@torch.no_grad()
def embed_images(images: list[Image.Image]) -> np.ndarray:
    """Return an (N, 512) float32 matrix of normalized image embeddings."""
    model, processor = _load()
    inputs = processor(images=images, return_tensors="pt")
    feats = model.get_image_features(**inputs)
    return F.normalize(feats, dim=-1).cpu().numpy().astype(np.float32)


@torch.no_grad()
def embed_text(text: str) -> np.ndarray:
    """Return a (512,) float32 normalized text embedding (truncated to CLIP's 77 tokens)."""
    model, processor = _load()
    inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True, max_length=77)
    feats = model.get_text_features(**inputs)
    return F.normalize(feats, dim=-1).cpu().numpy().flatten().astype(np.float32)
