# Multimodal Photo Search: CLIP + Claude Vision

Search a photo library in plain English ("person with a horse in a white polo") and get the three best matches with a written explanation for each. This is a multimodal RAG pipeline: a fast embedding search narrows thousands of photos to 10 candidates, then a vision-language model looks at those candidates and picks the best 3.

```
                 ┌──────────── offline, once ─────────────┐
 photos ───────► │ CLIP image embeddings  +  BLIP captions │──► FAISS index
                 └────────────────────────────────────────┘
                                                               │
 "dog playing                                                  ▼
  with a ball" ──► CLIP text embedding ──► Stage 1: top-10 candidates (~20 ms)
                                                               │
                                                               ▼
                                          Stage 2: Claude Vision reranks → top 3 + reasons (~5 s)
```

## Why two stages

| | Speed | Understanding |
|---|---|---|
| CLIP + FAISS alone | milliseconds over thousands of photos | good at topics, weak on fine detail (colors, counts, spatial relations) |
| Vision LLM alone | too slow and costly to scan every photo | strong visual reasoning |
| **Both** | **~20 ms to narrow, then one API call over 10 images** | **reasoning where it matters** |

## Results

Top-3 retrieval accuracy on a **~1,000-photo test set** drawn from the 5,400+ photo index:

| System | Top-3 hit rate |
|---|---|
| CLIP + FAISS only | ~60% |
| **CLIP + FAISS → Claude Vision rerank** | **85%** |

Latency (measured during development): ~20 ms for stage 1 over 5,400+ photos, ~5 s for stage 2.

**How these numbers were obtained.** Each test photo was labeled by a separate GPT model (ChatGPT), which wrote a natural-language description of it. Those descriptions served as the search queries, and a query counted as a hit when the photo it was written from appeared in the system's top 3. Because the labels come from a different model family than the reranker, they are independent of the system under test.

**Caveats.**
- These results were measured on an earlier version of the pipeline. The current reranking prompt no longer includes photo filenames and shuffles the candidates before showing them to the model, so a fresh run may produce somewhat different numbers.
- Queries are written from the target photo itself, so they are more specific than typical user searches.
- [`scripts/make_testset.py`](scripts/make_testset.py) and [`scripts/evaluate.py`](scripts/evaluate.py) implement this same protocol so anyone (including me) can reproduce it; see [Evaluation](#evaluation).

## Quick start

```bash
git clone https://github.com/abelsaj2005/multimodal-photo-search && cd multimodal-photo-search
pip install -r requirements.txt
cp .env.example .env        # add ANTHROPIC_API_KEY

# 1. put photos in data/photos/ (not included, see "Data")
python scripts/ingest.py --photos data/photos          # CLIP embeddings + BLIP captions -> data/index/

# 2. search
python scripts/search.py "person with a horse wearing a white polo"
```

Use `--no-captions` on ingest to skip BLIP (much faster; captions are metadata only and are not used for retrieval).

## Design decisions

- **CLIP over a fixed-class detector (e.g. YOLO).** CLIP maps text and images into the same space, so arbitrary queries work zero-shot with no training and no label set.
- **Cosine similarity via inner product on unit vectors** (`IndexFlatIP`). Exact search is fast enough at this scale; swap in an IVF/HNSW index for millions of photos.
- **Reranker sees shuffled candidates and no filenames.** Otherwise CLIP's ordering and names like `dog.66.jpg` would leak the answer to the model and inflate results.
- **Structured JSON output** from the reranker (`rank`, `photo_index`, `reason`) so results are machine-readable and every pick comes with an explanation.
- **Sonnet-class model for stage 2**, set with `FINDER_MODEL`. A single call per query keeps cost to roughly $0.05–0.10.

## Evaluation

```bash
export OPENAI_API_KEY=...   # only to write the test queries
python scripts/make_testset.py --n 1000      # GPT writes one query per sampled photo -> eval/testset.jsonl
python scripts/evaluate.py --limit 100       # smoke test first (~$5–10 of API calls)
python scripts/evaluate.py                   # full run; writes eval/results.json
python scripts/evaluate.py --no-rerank       # CLIP-only baseline, free
```

Limitations: queries are written from the target photo itself, so they are more specific than typical user searches; a near-duplicate of the target counts as a miss; results depend on the photo collection and reranker model version.

## Data

Photos are not redistributed. The experiments used a collection of 5,400+ photos (dogs, cats, horses with riders, and general scenes). Put any images (jpg, png, webp, bmp) in `data/photos/`.

## Project layout

```
finder/        embeddings.py (CLIP) · captioner.py (BLIP) · index.py (FAISS) · rerank.py (Claude) · search.py
scripts/       ingest.py · search.py · make_testset.py · evaluate.py
eval/          results.json and testset.jsonl (generated by the evaluation scripts, gitignored)
```

## Tech

Python · PyTorch · Hugging Face Transformers (CLIP ViT-B/32, BLIP-base) · FAISS · Anthropic API (Claude Vision) · OpenAI API (evaluation labels only)

## License

MIT
