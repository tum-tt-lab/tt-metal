# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""ImageNet-1k input pipeline for the "Better plain ViT baselines" recipe (Beyer et al. 2022).

Two layouts of <data_dir> are read:

- packed (made by prepare_imagenet.py from the Hugging Face copy; 6 files, random access by index)::

    <data_dir>/{train,validation}.bin, *_offsets.npy, *_labels.npy

- ImageFolder (class folders named by WordNet id, so sorted order gives the standard 0..999 labels)::

    <data_dir>/train/n01440764/*.JPEG
    <data_dir>/val/n01440764/*.JPEG      # val sorted into class folders (e.g. with valprep.sh)

Splits as in the paper: ``train`` = 99% of the train set, ``minival`` = the other 1% (picked with a
fixed seed, not the tail, since ImageFolder order is sorted by class), ``val`` = the official val set.

Augmentation (big_vision ``decode_jpeg_and_inception_crop(224)|flip_lr|randaug(2,10)``):

- inception crop: RandomResizedCrop(224, scale=(0.05, 1)) (big_vision ``area_min=5``)
- horizontal flip
- RandAugment(2 ops, magnitude 10 of 10) with grey (128) fill. torchvision's op set differs a bit
  from big_vision's TF implementation; magnitude is mapped so that 10 is the maximum, as there.

Eval: resize shorter side to 256, center crop 224. Images come out as uint8 (B, 3, 224, 224);
:func:`to_float` applies big_vision's ``value_range(-1, 1)``. Mixup happens on the host in the
training script, since it needs the batch.
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Sampler, Subset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

MINIVAL_FRACTION = 0.01
SPLIT_SEED = 0
IMAGENET_TRAIN_SIZE = 1_281_167
# Images per epoch after holding out minival; also the epoch size of --synthetic runs.
TRAIN_SPLIT_SIZE = IMAGENET_TRAIN_SIZE - int(round(IMAGENET_TRAIN_SIZE * MINIVAL_FRACTION))


def train_transform(image_size: int, randaug_ops: int, randaug_magnitude: int) -> transforms.Compose:
    ops = [
        transforms.RandomResizedCrop(image_size, scale=(0.05, 1.0), interpolation=InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(),
    ]
    if randaug_ops > 0:
        # num_magnitude_bins=11 -> magnitude 10 is the strongest setting, like big_vision's _MAX_LEVEL=10.
        ops.append(
            transforms.RandAugment(
                num_ops=randaug_ops, magnitude=randaug_magnitude, num_magnitude_bins=11, fill=[128, 128, 128]
            )
        )
    ops.append(transforms.PILToTensor())
    return transforms.Compose(ops)


def eval_transform(image_size: int, resize: int = 256) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(resize, interpolation=InterpolationMode.BILINEAR),
            transforms.CenterCrop(image_size),
            transforms.PILToTensor(),
        ]
    )


def _split_indices(n: int) -> tuple[np.ndarray, np.ndarray]:
    perm = np.random.default_rng(SPLIT_SEED).permutation(n)
    n_minival = int(round(n * MINIVAL_FRACTION))
    return np.sort(perm[n_minival:]), np.sort(perm[:n_minival])


class PackedImages(torch.utils.data.Dataset):
    """One split written by prepare_imagenet.py: JPEG i = <split>.bin[offsets[i]:offsets[i + 1]].

    The file is opened lazily in each DataLoader worker (after fork) and read with one pread per image.
    """

    def __init__(self, data_dir: Path, split: str, transform) -> None:
        self.path = data_dir / f"{split}.bin"
        self.offsets = np.load(data_dir / f"{split}_offsets.npy")
        self.labels = np.load(data_dir / f"{split}_labels.npy").astype(np.int64)
        self.transform = transform
        self._fd, self._pid = None, None

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        if self._pid != os.getpid():
            self._fd, self._pid = os.open(self.path, os.O_RDONLY), os.getpid()
        start, end = int(self.offsets[i]), int(self.offsets[i + 1])
        img = Image.open(io.BytesIO(os.pread(self._fd, end - start, start))).convert("RGB")
        return self.transform(img), int(self.labels[i])


def build_datasets(
    data_dir: Path, image_size: int = 224, randaug_ops: int = 2, randaug_magnitude: int = 10
) -> dict[str, torch.utils.data.Dataset]:
    """{'train', 'minival', 'val'} datasets yielding (uint8 CHW tensor, int label)."""
    train_tf = train_transform(image_size, randaug_ops, randaug_magnitude)
    if (data_dir / "train.bin").exists():
        train_full = PackedImages(data_dir, "train", train_tf)
        train_idx, minival_idx = _split_indices(len(train_full))
        return {
            "train": Subset(train_full, train_idx.tolist()),
            "minival": Subset(PackedImages(data_dir, "train", eval_transform(image_size)), minival_idx.tolist()),
            "val": PackedImages(data_dir, "validation", eval_transform(image_size)),
        }
    train_dir, val_dir = data_dir / "train", data_dir / "val"
    for d in (train_dir, val_dir):
        if not d.is_dir():
            raise FileNotFoundError(f"ImageNet folder {d} not found (expected <data_dir>/train and <data_dir>/val)")
    train_full = datasets.ImageFolder(train_dir, transform=train_tf)
    train_eval_view = datasets.ImageFolder(train_dir, transform=eval_transform(image_size))
    val = datasets.ImageFolder(val_dir, transform=eval_transform(image_size))
    if train_full.classes != val.classes:
        raise ValueError(f"train/val class folders differ ({len(train_full.classes)} vs {len(val.classes)})")
    train_idx, minival_idx = _split_indices(len(train_full))
    return {
        "train": Subset(train_full, train_idx.tolist()),
        "minival": Subset(train_eval_view, minival_idx.tolist()),
        "val": val,
    }


class EpochShuffle(Sampler[int]):
    """Permutation seeded by (seed, epoch): the same order for an epoch whether or not the run was resumed.

    (DataLoader(shuffle=True) with a reseeded generator is not: with persistent workers the generator is
    also drawn from when the first iterator is created, so the first epoch after a resume differs.)
    """

    def __init__(self, n: int, seed: int) -> None:
        self.n, self.seed, self.epoch = n, seed, 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        return iter(np.random.default_rng([self.seed, self.epoch]).permutation(self.n).tolist())

    def __len__(self) -> int:
        return self.n


def make_loader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    workers: int,
    sampler: EpochShuffle | None = None,
) -> DataLoader:
    """Train loader when `sampler` is given (drops the last partial batch), else an in-order eval loader."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=sampler is not None,
        num_workers=workers,
        persistent_workers=workers > 0,
        prefetch_factor=2 if workers > 0 else None,  # batches are whole steps (1024 images)
    )


def to_float(x_uint8: np.ndarray) -> np.ndarray:
    """uint8 images -> float32 in [-1, 1] (big_vision ``value_range(-1, 1)``)."""
    return x_uint8.astype(np.float32) / 127.5 - 1.0
