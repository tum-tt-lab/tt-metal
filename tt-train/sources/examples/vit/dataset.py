# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""CIFAR-10 (binary version) loading and numpy augmentation.

Format: https://www.cs.toronto.edu/~kriz/cifar.html ("CIFAR-10 binary version").
Each record is 1 label byte followed by 3072 bytes of a 3x32x32 image in CHW order.
"""

from __future__ import annotations

import os
import tarfile
import urllib.request
from pathlib import Path

import numpy as np

URL = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(1, 3, 1, 1)
STD = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32).reshape(1, 3, 1, 1)
RECORD = 1 + 3 * 32 * 32


def default_data_dir() -> Path:
    cache = os.environ.get("TT_CACHE") or os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(cache) / "datasets" / "cifar-10-batches-bin"


def download(data_dir: Path) -> None:
    """Download and extract CIFAR-10 so that data_dir/data_batch_1.bin exists (needs internet)."""
    data_dir.parent.mkdir(parents=True, exist_ok=True)
    archive = data_dir.parent / "cifar-10-binary.tar.gz"
    if not archive.exists():
        print(f"downloading {URL} -> {archive}")
        urllib.request.urlretrieve(URL, archive)
    with tarfile.open(archive) as tar:
        tar.extractall(data_dir.parent, filter="data")  # extracts cifar-10-batches-bin/
    print(f"extracted to {data_dir}")


def _read(files: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    raw = np.concatenate([np.fromfile(f, dtype=np.uint8).reshape(-1, RECORD) for f in files])
    return raw[:, 1:].reshape(-1, 3, 32, 32), raw[:, 0].astype(np.uint32)


def load(data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return x_train (50000, 3, 32, 32) uint8, y_train (50000,), x_test (10000, ...), y_test (10000,)."""
    if not (data_dir / "data_batch_1.bin").exists():
        raise FileNotFoundError(
            f"CIFAR-10 not found in {data_dir}. On a node with internet run:\n"
            f"  python tt-train/sources/examples/vit/dataset.py --data-dir {data_dir}"
        )
    x_train, y_train = _read([data_dir / f"data_batch_{i}.bin" for i in range(1, 6)])
    x_test, y_test = _read([data_dir / "test_batch.bin"])
    return x_train, y_train, x_test, y_test


def normalize(x: np.ndarray) -> np.ndarray:
    """uint8 (B, 3, 32, 32) -> float32, per-channel standardised."""
    return (x.astype(np.float32) / 255.0 - MEAN) / STD


def augment(x: np.ndarray, rng: np.random.Generator, pad: int = 4) -> np.ndarray:
    """Random crop (zero pad `pad`, crop back to 32x32) + random horizontal flip, on uint8 (B, 3, 32, 32)."""
    B, _, H, W = x.shape
    padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    dy = rng.integers(0, 2 * pad + 1, B)
    dx = rng.integers(0, 2 * pad + 1, B)
    rows = dy[:, None] + np.arange(H)[None, :]  # (B, H)
    cols = dx[:, None] + np.arange(W)[None, :]  # (B, W)
    out = padded[
        np.arange(B)[:, None, None, None],
        np.arange(3)[None, :, None, None],
        rows[:, None, :, None],
        cols[:, None, None, :],
    ]
    flip = rng.random(B) < 0.5
    out[flip] = out[flip, :, :, ::-1]
    return out


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Download CIFAR-10 (binary) for the ViT example")
    p.add_argument("--data-dir", type=Path, default=default_data_dir())
    args = p.parse_args()
    if (args.data_dir / "data_batch_1.bin").exists():
        print(f"already present: {args.data_dir}")
    else:
        download(args.data_dir)
