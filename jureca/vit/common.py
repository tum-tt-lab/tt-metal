"""Everything the PyTorch and the Tenstorrent version must share, so both see the same data in the same order."""

import argparse
import math
import os
import pickle

import torch

IMAGE_SIZE = 32  # property of CIFAR-100
NUM_CLASSES = 100

MODELS = {  # come from DeiT-Ti. Table 1 of the Paper https://arxiv.org/abs/2012.12877
    "tiny": dict(depth=12, dim=192, heads=3, mlp_dim=768),
    "small": dict(depth=12, dim=384, heads=6, mlp_dim=1536),
    "base": dict(depth=12, dim=768, heads=12, mlp_dim=3072),
}
PATCH_SIZE = 4
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2  # 8 x 8 = 64 tokens


# The ImageNet values, which DeiT also uses for CIFAR (DeiT datasets.py L108).
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

WARMUP_START_LR = 1e-6  # DeiT main.py L79
MIN_LR = 1e-5  # DeiT main.py L81


def load_cifar100(data_dir, split):
    """Returns images (N, 3, 32, 32) as normalized float32 and labels (N,) as int64."""
    with open(os.path.join(data_dir, split), "rb") as f:
        d = pickle.load(f, encoding="latin1")
    images = torch.from_numpy(d["data"]).view(-1, 3, IMAGE_SIZE, IMAGE_SIZE).float() / 255.0
    images = (images - MEAN) / STD
    labels = torch.tensor(d["fine_labels"], dtype=torch.int64)
    return images, labels


def batches(num_images, batch_size, generator):
    """Index tensors for one epoch: new random order, last incomplete batch dropped (97 batches of 512)."""
    order = torch.randperm(num_images, generator=generator)
    for i in range(num_images // batch_size):
        # same as return, but continues where it left off in the loop
        yield order[i * batch_size : (i + 1) * batch_size]


def parse_args(default_out):
    """Settings shared by both versions; the defaults are the baseline recipe (README.md)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="tiny", choices=MODELS)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-4, help="for batch size 512; scaled with the batch size")
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=None, help="stop early (smoke tests)")
    parser.add_argument("--images", type=int, default=None, help="use only the first N images (smoke tests)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--data", default=os.path.join(os.environ.get("SCRATCH", "."), "merrath1/data/cifar-100-python")
    )  # Data is loaded on SCRATCH
    parser.add_argument("--out", default=default_out)
    return parser.parse_args()


def lr_at(epoch, epochs, warmup_epochs, peak_lr):
    """Learning rate of one epoch (constant within the epoch), as timm's cosine schedule in DeiT:
    linear warmup from WARMUP_START_LR, then cosine decay towards MIN_LR."""
    if epoch < warmup_epochs:
        return WARMUP_START_LR + epoch * (peak_lr - WARMUP_START_LR) / warmup_epochs
    return MIN_LR + 0.5 * (peak_lr - MIN_LR) * (1 + math.cos(math.pi * epoch / epochs))


class Log:
    """steps.csv: loss of every step (the curve to reproduce). log.csv: one row per epoch.
    Opening with "w" overwrites an earlier run in the same output folder."""

    def __init__(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)  # creates the out dir, if non existent
        self.steps = open(os.path.join(out_dir, "steps.csv"), "w")
        self.steps.write("step,loss,lr\n")
        self.epochs = open(os.path.join(out_dir, "log.csv"), "w")
        self.epochs.write("epoch,step,train_loss,test_loss,test_acc,images_per_s\n")

    def step(self, step, loss, lr):
        self.steps.write(f"{step},{loss:.6f},{lr:.6e}\n")
        if step % 10 == 0:
            print(f"  step {step:6d}  loss {loss:.4f}  lr {lr:.2e}")

    def epoch(self, epoch, step, train_loss, test_loss, test_acc, images_per_s):
        self.epochs.write(f"{epoch},{step},{train_loss:.4f},{test_loss:.4f},{test_acc:.4f},{images_per_s:.0f}\n")
        print(
            f"epoch {epoch:3d}  train_loss {train_loss:.4f}  test_loss {test_loss:.4f}  "
            f"test_acc {test_acc:.4f}  {images_per_s:.0f} img/s"
        )
        self.steps.flush()  # write to disk now, so a crashed or cancelled job keeps its log
        self.epochs.flush()
