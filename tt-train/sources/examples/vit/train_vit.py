# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Train a ViT from scratch on CIFAR-10 with ttml on a single device.

Example (1 chip, smoke test):
    python tt-train/sources/examples/vit/train_vit.py --max-steps 50 --eval-batches 2
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np
import ttnn
import ttml
from ttml.common.utils import no_grad, set_seed
from ttml.models.vit import ViT, ViTConfig, patchify

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    m = p.add_argument_group("model (defaults: ViT-Tiny, 4x4 patches on 32x32 images)")
    m.add_argument("--patch-size", type=int, default=4)
    m.add_argument("--dim", type=int, default=192)
    m.add_argument("--heads", type=int, default=3)
    m.add_argument("--depth", type=int, default=12)
    m.add_argument("--dropout", type=float, default=0.1)
    t = p.add_argument_group("training")
    t.add_argument("--data-dir", type=Path, default=dataset.default_data_dir())
    t.add_argument("--batch-size", type=int, default=128)
    t.add_argument("--epochs", type=int, default=100)
    t.add_argument("--max-steps", type=int, default=0, help="> 0 stops after this many steps")
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--min-lr-factor", type=float, default=0.01)
    t.add_argument("--warmup-epochs", type=float, default=5.0)
    t.add_argument("--weight-decay", type=float, default=0.05)
    t.add_argument("--clip-grad-norm", type=float, default=1.0, help="0 disables clipping")
    t.add_argument("--no-augment", action="store_true")
    t.add_argument("--seed", type=int, default=42)
    o = p.add_argument_group("eval / logging")
    o.add_argument("--eval-every", type=int, default=1, help="evaluate every N epochs")
    o.add_argument("--eval-batches", type=int, default=0, help="> 0 limits eval to N batches")
    o.add_argument("--log-every", type=int, default=20)
    o.add_argument("--metrics", type=Path, default=None, help="CSV file for step/loss/lr/accuracy")
    return p.parse_args()


def to_device(x_uint8: np.ndarray, y: np.ndarray, config: ViTConfig):
    patches = patchify(dataset.normalize(x_uint8), config)
    tt_x = ttml.autograd.Tensor.from_numpy(patches, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16)
    tt_y = ttml.autograd.Tensor.from_numpy(
        y.reshape(-1, 1).astype(np.uint32), ttnn.Layout.ROW_MAJOR, ttnn.DataType.UINT32
    )
    return tt_x, tt_y


def host_logits(logits: ttml.autograd.Tensor, config: ViTConfig) -> np.ndarray:
    return logits.to_numpy().astype(np.float32).reshape(-1, config.num_classes)


def host_cross_entropy(logits: np.ndarray, y: np.ndarray) -> float:
    """Mean CE in fp32 on the host. The device loss is a bf16 scalar whose batch mean is too coarse to log."""
    z = logits - logits.max(axis=1, keepdims=True)
    log_probs = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    return float(-log_probs[np.arange(len(y)), y].mean())


def lr_at(step: int, total: int, warmup: int, base: float, min_factor: float) -> float:
    """Linear warmup, then cosine decay to base * min_factor."""
    if step < warmup:
        return base * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return base * (min_factor + (1 - min_factor) * 0.5 * (1 + math.cos(math.pi * progress)))


def evaluate(
    model: ViT, x: np.ndarray, y: np.ndarray, config: ViTConfig, batch_size: int, max_batches: int
) -> tuple[float, float]:
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    with no_grad():
        for i, start in enumerate(range(0, len(x), batch_size)):
            if max_batches and i >= max_batches:
                break
            xb, yb = x[start : start + batch_size], y[start : start + batch_size]
            tt_x, _ = to_device(xb, yb, config)
            logits = host_logits(model(tt_x), config)
            correct += int((logits.argmax(axis=1) == yb).sum())
            loss_sum += host_cross_entropy(logits, yb) * len(xb)
            total += len(xb)
    model.train()
    return loss_sum / total, correct / total


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    x_train, y_train, x_test, y_test = dataset.load(args.data_dir)
    print(f"CIFAR-10: train {x_train.shape}, test {x_test.shape} from {args.data_dir}")

    ctx = ttml.autograd.AutoContext.get_instance()
    ctx.open_device()
    try:
        config = ViTConfig(
            patch_size=args.patch_size,
            embedding_dim=args.dim,
            num_heads=args.heads,
            num_blocks=args.depth,
            dropout=args.dropout,
        )
        model = ViT(config)
        params = model.parameters()
        n_params = sum(int(np.prod(t.shape())) for t in params.values())
        print(
            f"ViT: {config}\n  tokens {config.seq_len} (padded {config.padded_seq_len}), "
            f"patch_dim {config.patch_dim}, params {n_params / 1e6:.2f}M"
        )

        optimizer = ttml.optimizers.create_optimizer(
            {
                "type": "AdamW",
                "lr": args.lr,
                "beta1": 0.9,
                "beta2": 0.999,
                "epsilon": 1e-8,
                "weight_decay": args.weight_decay,
                # No weight decay on biases, LayerNorm and position embeddings.
                "param_groups": [{"modules": ["bias", "gamma", "beta", "pos_emb"], "weight_decay": 0.0}],
            },
            params,
        )

        steps_per_epoch = len(x_train) // args.batch_size  # drop the last partial batch
        total_steps = args.max_steps or args.epochs * steps_per_epoch
        warmup_steps = int(args.warmup_epochs * steps_per_epoch)
        print(f"{total_steps} steps ({steps_per_epoch}/epoch), batch {args.batch_size}, warmup {warmup_steps}")

        metrics = None
        if args.metrics:
            args.metrics.parent.mkdir(parents=True, exist_ok=True)
            metrics = csv.writer(
                open(args.metrics, "w", newline="", buffering=1)
            )  # line-buffered: rows survive a timeout
            metrics.writerow(["step", "epoch", "lr", "train_loss", "test_loss", "test_acc", "img_per_s"])

        model.train()
        step, epoch, last_log_step = 0, 0, 0
        t0 = time.time()
        while step < total_steps:
            perm = rng.permutation(len(x_train))
            for b in range(steps_per_epoch):
                if step >= total_steps:
                    break
                idx = perm[b * args.batch_size : (b + 1) * args.batch_size]
                xb = x_train[idx] if args.no_augment else dataset.augment(x_train[idx], rng)
                tt_x, tt_y = to_device(xb, y_train[idx], config)

                lr = lr_at(step, total_steps, warmup_steps, args.lr, args.min_lr_factor)
                optimizer.set_lr(lr)
                optimizer.zero_grad()
                logits = model(tt_x)
                loss = ttml.ops.loss.cross_entropy_loss(logits, tt_y, ttml.ops.ReduceType.MEAN)
                loss.backward(False)
                if args.clip_grad_norm > 0:
                    ttml.core.clip_grad_norm(params, args.clip_grad_norm, 2.0, False)
                optimizer.step()
                ctx.reset_graph()
                step += 1

                if step % args.log_every == 0 or step == 1:
                    loss_val = host_cross_entropy(host_logits(logits, config), y_train[idx])
                    device_loss = float(loss.to_numpy().item())
                    ips = (step - last_log_step) * args.batch_size / (time.time() - t0) if step > 1 else 0.0
                    t0, last_log_step = time.time(), step
                    print(
                        f"step {step:6d}  epoch {epoch}  lr {lr:.2e}  loss {loss_val:.4f} (device bf16 {device_loss:.4f})  {ips:7.1f} img/s",
                        flush=True,
                    )
                    if metrics:
                        metrics.writerow([step, epoch, lr, loss_val, "", "", ips])

            epoch += 1
            if epoch % args.eval_every == 0 or step >= total_steps:
                test_loss, test_acc = evaluate(model, x_test, y_test, config, args.batch_size, args.eval_batches)
                print(
                    f"== epoch {epoch}  step {step}  test loss {test_loss:.4f}  test acc {test_acc * 100:.2f}%",
                    flush=True,
                )
                if metrics:
                    metrics.writerow([step, epoch, "", "", test_loss, test_acc, ""])
                t0, last_log_step = time.time(), step
    finally:
        ctx.close_device()


if __name__ == "__main__":
    main()
