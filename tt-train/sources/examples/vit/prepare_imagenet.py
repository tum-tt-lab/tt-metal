# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Download ImageNet-1k (Hugging Face ``ILSVRC/imagenet-1k``) and pack it for random access.

    python prepare_imagenet.py download --hf-dir DIR      # needs `hf auth login` + accepted terms
    python prepare_imagenet.py pack --hf-dir DIR --out OUT

The Hub copy is Parquet (train 294 shards, validation 14; row groups of ~750 images), which cannot be
read one image at a time. ``pack`` rewrites each split as three files, in shard order::

    OUT/<split>.bin          all JPEG bytes back to back
    OUT/<split>_offsets.npy  int64 (N + 1,): image i is bin[offsets[i]:offsets[i + 1]]
    OUT/<split>_labels.npy   int16 (N,), the standard 0..999 class index

That is 6 files instead of 1.33M JPEGs (PROJECT and SCRATCH have file-count quotas), and imagenet.py
reads one image with a single pread. Splits: train (1,281,167) and validation (50,000).
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

REPO = "ILSVRC/imagenet-1k"
SPLITS = {"train": 1_281_167, "validation": 50_000}


def download(hf_dir: Path, workers: int) -> None:
    from huggingface_hub import snapshot_download

    t = time.time()
    snapshot_download(
        REPO,
        repo_type="dataset",
        local_dir=hf_dir,
        allow_patterns=[f"data/{s}-*.parquet" for s in SPLITS],
        max_workers=workers,
    )
    print(f"downloaded to {hf_dir} in {(time.time() - t) / 60:.1f} min", flush=True)


def pack_split(hf_dir: Path, out: Path, split: str) -> None:
    import pyarrow.parquet as pq

    shards = sorted((hf_dir / "data").glob(f"{split}-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no {split} shards in {hf_dir / 'data'}")
    n_shards = int(shards[0].stem.rsplit("-of-", 1)[1])
    if len(shards) != n_shards:
        raise FileNotFoundError(f"{split}: {len(shards)} of {n_shards} shards present; download incomplete")
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / f"{split}.bin.tmp"
    offsets, labels, pos, t = [0], [], 0, time.time()
    with open(tmp, "wb") as f:
        for k, shard in enumerate(shards):
            pf = pq.ParquetFile(shard)
            for g in range(pf.num_row_groups):
                tbl = pf.read_row_group(g, columns=["image", "label"])
                images = tbl.column("image").combine_chunks().field("bytes")
                for b in images.to_pylist():
                    f.write(b)
                    pos += len(b)
                    offsets.append(pos)
                labels.extend(tbl.column("label").to_pylist())
            if k % 20 == 0 or k == len(shards) - 1:
                gb = pos / 1e9
                print(f"{split}: shard {k + 1}/{len(shards)}, {len(labels)} images, {gb:.1f} GB, "
                      f"{gb / (time.time() - t) * 1e3:.0f} MB/s", flush=True)  # fmt: skip
    labels = np.asarray(labels, dtype=np.int16)
    if len(labels) != SPLITS[split] or labels.min() < 0 or labels.max() > 999:
        raise ValueError(f"{split}: {len(labels)} images (expected {SPLITS[split]}), labels {labels.min()}..{labels.max()}")
    np.save(out / f"{split}_offsets.npy", np.asarray(offsets, dtype=np.int64))
    np.save(out / f"{split}_labels.npy", labels)
    os.replace(tmp, out / f"{split}.bin")  # last, so a present .bin means the split is complete
    print(f"{split}: packed {len(labels)} images, {pos / 1e9:.1f} GB -> {out}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", choices=["download", "pack", "all"])
    p.add_argument("--hf-dir", type=Path, required=True, help="Parquet download directory")
    p.add_argument("--out", type=Path, help="packed output directory (pack/all)")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    if args.action in ("download", "all"):
        download(args.hf_dir, args.workers)
    if args.action in ("pack", "all"):
        if args.out is None:
            p.error("--out is required for pack")
        for split in SPLITS:
            if (args.out / f"{split}.bin").exists():
                print(f"{split}: already packed")
            else:
                pack_split(args.hf_dir, args.out, split)


if __name__ == "__main__":
    main()
