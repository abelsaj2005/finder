"""Persistent photo index: embeddings.npy + metadata.json, searched with FAISS."""
from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

DEFAULT_DIR = Path("data/index")


class PhotoIndex:
    def __init__(self, embeddings: np.ndarray, metadata: list[dict]):
        assert len(embeddings) == len(metadata), "embeddings and metadata must align"
        self.embeddings = embeddings.astype("float32")
        self.metadata = metadata
        # Embeddings are unit-length, so inner product == cosine similarity.
        self._faiss = faiss.IndexFlatIP(self.embeddings.shape[1])
        self._faiss.add(self.embeddings)

    def __len__(self) -> int:
        return len(self.metadata)

    def search(self, query_vec: np.ndarray, k: int = 10) -> list[dict]:
        scores, ids = self._faiss.search(query_vec.reshape(1, -1).astype("float32"), k)
        return [{**self.metadata[i], "score": float(s)} for s, i in zip(scores[0], ids[0])]

    def save(self, directory: Path = DEFAULT_DIR) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "embeddings.npy", self.embeddings)
        (directory / "metadata.json").write_text(json.dumps(self.metadata, indent=2))

    @classmethod
    def load(cls, directory: Path = DEFAULT_DIR) -> "PhotoIndex":
        emb = np.load(directory / "embeddings.npy")
        meta = json.loads((directory / "metadata.json").read_text())
        return cls(emb, meta)
