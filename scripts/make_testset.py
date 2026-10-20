"""Build an evaluation set: sample N indexed photos and have an *independent* model (GPT)
write a natural-language search query describing each one.

The target image for each query is the photo it was written from, so "hit" = the system
returns that photo. Using a different model family than the reranker (Claude) keeps the
labels independent of the system under test.

    export OPENAI_API_KEY=...
    python scripts/make_testset.py --n 1000 --out eval/testset.jsonl
"""
import argparse
import base64
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openai import OpenAI
from tqdm import tqdm

from finder.index import PhotoIndex

PROMPT = ("Write one natural-language search query (8-20 words) that someone might type to find "
          "this exact photo in a large gallery. Mention the distinguishing content (subject, action, "
          "colors, setting). Do not mention image quality or say 'photo' or 'image'. Return only the query.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--out", type=Path, default=Path("eval/testset.jsonl"))
    args = ap.parse_args()

    index = PhotoIndex.load()
    rng = random.Random(args.seed)
    sample = rng.sample(index.metadata, min(args.n, len(index)))
    client = OpenAI()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    with args.out.open("w") as f:
        for item in tqdm(sample):
            b64 = base64.b64encode(Path(item["image_path"]).read_bytes()).decode()
            ext = Path(item["image_path"]).suffix.lstrip(".").lower().replace("jpg", "jpeg")
            resp = client.chat.completions.create(
                model=args.model, max_tokens=60, temperature=0,
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/{ext};base64,{b64}"}},
                ]}],
            )
            f.write(json.dumps({"image_path": item["image_path"],
                                "query": resp.choices[0].message.content.strip()}) + "\n")
    print(f"Wrote {len(sample)} queries to {args.out}")


if __name__ == "__main__":
    main()
