"""Evaluate retrieval quality on eval/testset.jsonl.

Reports, for the same queries:
  * CLIP-only      : top-1 / top-3 hit rate of stage 1 alone
  * CLIP recall@10 : the ceiling for reranking (is the target even in the candidate set?)
  * Full system    : top-1 / top-3 hit rate after Claude Vision reranking
plus median stage-1 latency and mean stage-2 latency. Results go to eval/results.json.

    python scripts/evaluate.py --limit 100          # cheap smoke test first
    python scripts/evaluate.py                      # full test set (see README for cost)
    python scripts/evaluate.py --no-rerank          # CLIP-only, free
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tqdm import tqdm

from finder.index import PhotoIndex
from finder.search import search


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--testset", type=Path, default=Path("eval/testset.jsonl"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=Path("eval/results.json"))
    ap.add_argument("--no-rerank", action="store_true", help="CLIP-only (free, no API calls)")
    args = ap.parse_args()

    index = PhotoIndex.load()
    rows = [json.loads(line) for line in args.testset.read_text().splitlines() if line.strip()]
    if args.limit:
        rows = rows[: args.limit]

    clip1 = clip3 = clip10 = full1 = full3 = failures = 0
    s1_ms, s2_s = [], []

    for row in tqdm(rows):
        target = row["image_path"]
        # Stage 1 is always computed (it is also the CLIP-only baseline).
        base = search(index, row["query"], use_rerank=False)
        cpaths = [c["image_path"] for c in base["candidates"]]
        clip1 += cpaths[0] == target
        clip3 += target in cpaths[:3]
        clip10 += target in cpaths
        s1_ms.append(base["stage1_ms"])
        if args.no_rerank:
            continue

        try:
            out = search(index, row["query"])
            ranked = [r["image_path"] for r in sorted(out["results"], key=lambda r: r["rank"])]
        except Exception as e:  # API / JSON-parse errors count as a miss and are reported
            failures += 1
            print("rerank failure:", e)
            ranked = []
        full1 += bool(ranked) and ranked[0] == target
        full3 += target in ranked[:3]
        if ranked:
            s2_s.append(out["stage2_s"])

    n = len(rows)
    res = {
        "n_queries": n, "index_size": len(index),
        "clip_only_top1": clip1 / n, "clip_only_top3": clip3 / n, "clip_recall_at_10": clip10 / n,
        "median_stage1_ms": statistics.median(s1_ms),
    }
    if not args.no_rerank:
        res.update({
            "full_top1": full1 / n, "full_top3": full3 / n,
            "mean_stage2_s": statistics.mean(s2_s) if s2_s else None,
            "rerank_failures": failures,
        })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
