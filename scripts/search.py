"""Command-line demo.

    python scripts/search.py "person with a horse wearing a white polo"
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from finder import PhotoIndex, search


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--no-rerank", action="store_true")
    args = ap.parse_args()

    out = search(PhotoIndex.load(), args.query, use_rerank=not args.no_rerank)
    print(f"Stage 1 (CLIP + FAISS): {out['stage1_ms']:.1f} ms")
    if not args.no_rerank:
        print(f"Stage 2 (Claude Vision): {out['stage2_s']:.1f} s")
    for r in out["results"]:
        print(f"\n#{r.get('rank', '-')} {r['image_path']}")
        if "reason" in r:
            print(f"   {r['reason']}")


if __name__ == "__main__":
    main()
