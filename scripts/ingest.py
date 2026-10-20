"""Index a folder of photos: CLIP embeddings + BLIP captions -> data/index/.

    python scripts/ingest.py --photos data/photos [--no-captions]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image
from tqdm import tqdm

from finder.embeddings import embed_images
from finder.index import DEFAULT_DIR, PhotoIndex

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photos", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=DEFAULT_DIR)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--no-captions", action="store_true", help="skip BLIP captioning (much faster)")
    args = ap.parse_args()

    paths = sorted(p for p in args.photos.rglob("*") if p.suffix.lower() in EXTS)
    if not paths:
        sys.exit(f"No images found in {args.photos}")
    print(f"Indexing {len(paths)} photos...")

    if not args.no_captions:
        from finder.captioner import caption

    embs, meta = [], []
    for i in tqdm(range(0, len(paths), args.batch)):
        batch_paths = paths[i:i + args.batch]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        embs.append(embed_images(images))
        for p, img in zip(batch_paths, images):
            meta.append({
                "image_path": str(p),
                "filename": p.name,
                "caption": "" if args.no_captions else caption(img),
            })

    PhotoIndex(np.vstack(embs), meta).save(args.out)
    print(f"Saved {len(meta)} photos to {args.out}/")


if __name__ == "__main__":
    main()
