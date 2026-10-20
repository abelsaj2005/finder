"""Stage 2: rerank CLIP candidates with Claude Vision."""
from __future__ import annotations

import base64
import json
import os
import random
import re
from pathlib import Path

import anthropic

MODEL = os.environ.get("FINDER_MODEL", "claude-sonnet-4-6")
MEDIA_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}

PROMPT = """You are given {n} photos. Pick the {k} that best match this search query:

Query: "{query}"

Return ONLY a JSON array of exactly {k} objects, best match first:
[{{"rank": 1, "photo_index": 3, "reason": "short explanation"}}]

Photos:"""


def _client() -> anthropic.Anthropic:
    return anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment


def rerank(query: str, candidates: list[dict], final_k: int = 3, seed: int = 0) -> list[dict]:
    """Return the top `final_k` candidates with Claude's explanation.

    Candidates are shuffled before being shown to the model (so CLIP's ordering cannot leak
    into the ranking), and filenames are never included in the prompt (so names like
    `dog.66.jpg` cannot leak the answer).
    """
    order = list(range(len(candidates)))
    random.Random(seed).shuffle(order)

    content = [{"type": "text", "text": PROMPT.format(n=len(order), k=final_k, query=query)}]
    for shown_idx, cand_idx in enumerate(order):
        path = Path(candidates[cand_idx]["image_path"])
        data = base64.standard_b64encode(path.read_bytes()).decode()
        content.append({"type": "text", "text": f"[Photo {shown_idx}]"})
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": MEDIA_TYPES.get(path.suffix.lower(), "image/jpeg"), "data": data},
        })

    resp = _client().messages.create(
        model=MODEL, max_tokens=1024, temperature=0,
        messages=[{"role": "user", "content": content}],
    )
    text = resp.content[0].text
    match = re.search(r"\[\s*\{.*\}\s*\]", text, re.DOTALL)
    parsed = json.loads(match.group(0) if match else text)

    results = []
    for item in parsed[:final_k]:
        shown = int(item["photo_index"])
        if 0 <= shown < len(order):
            results.append({**candidates[order[shown]], "rank": int(item["rank"]), "reason": item["reason"]})
    return results
