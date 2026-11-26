# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Train a ViT from scratch on CIFAR-10 with ttml, on one device or data-parallel (DDP) on several.

DDP (--num-devices N > 1): the devices form a 1xN mesh. Parameters are replicated, every global
batch of N * --batch-size images is split along the batch dim (each device gets --batch-size), and
after backward the gradients are averaged over the mesh (all-reduce over the Ethernet fabric), so
every device applies the same AdamW update. Needs TT_MESH_GRAPH_DESC_PATH pointing at a 1xN mesh
graph descriptor whose devices are Ethernet-linked (train_vit.sbatch with NDEV=N sets this up).

Example (1 chip, smoke test):
    python tt-train/sources/examples/vit/train_vit.py --max-steps 50 --eval-batches 2
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import ttnn
import ttml
from ttml.common.run_logger import RunLogger
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
    t.add_argument("--batch-size", type=int, default=128, help="per device; global batch = this * --num-devices")
    t.add_argument("--num-devices", type=int, default=1, help="> 1: data-parallel over a 1xN device mesh")
    t.add_argument(
        "--grad-sync",
        choices=["bucketed", "per-param"],
        default="bucketed",
        help="DDP gradient all-reduce: one per group of same-shape parameters, or one per parameter",
    )
    t.add_argument("--epochs", type=int, default=100)
    t.add_argument("--max-steps", type=int, default=0, help="> 0 stops after this many steps")
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--min-lr-factor", type=float, default=0.01)
    t.add_argument("--warmup-epochs", type=float, default=5.0)
    t.add_argument("--weight-decay", type=float, default=0.05)
    t.add_argument("--clip-grad-norm", type=float, default=1.0, help="0 disables clipping")
    t.add_argument("--no-augment", action="store_true")
    t.add_argument(
        "--loader-workers",
        type=int,
        default=4,
        help="host threads that augment + patchify upcoming batches while the device trains (0: inline)",
    )
    t.add_argument("--seed", type=int, default=42)
    o = p.add_argument_group("eval / logging")
    o.add_argument("--eval-every", type=int, default=1, help="evaluate every N epochs")
    o.add_argument("--eval-batches", type=int, default=0, help="> 0 limits eval to N batches")
    o.add_argument("--log-every", type=int, default=20)
    o.add_argument("--metrics", type=Path, default=None, help="CSV file for step/loss/lr/accuracy")
    o.add_argument("--wandb-project", default=None, help="log to this wandb project (off if unset)")
    o.add_argument("--wandb-name", default=None, help="wandb run name (default: vit-<slurm job id>)")
    return p.parse_args()


METRIC_COLUMNS = [
    "epoch",
    "images_seen",
    "train/loss",
    "train/loss_device_bf16",
    "train/lr",
    "train/img_per_s",
    "train/grad_sync_ms",
    "test/loss",
    "test/acc",
]


class Devices:
    """The opened device(s) plus the host<->mesh batch mapping.

    One device: plain open_device, tensors are copied as they are.
    N devices: a 1xN mesh with DDP; batch tensors are sharded along dim 0 (mapper) and outputs are
    concatenated back along dim 0 (composer). Tensors created without a mapper (parameters, the
    attention mask) are replicated on every device.
    """

    DP_AXIS = 1  # mesh (1, N): the data-parallel axis

    def __init__(self, num_devices: int, grad_sync: str = "bucketed") -> None:
        self.n = num_devices
        self.grad_sync = grad_sync
        self.ctx = ttml.autograd.AutoContext.get_instance()
        if self.n > 1:
            ttml.open_device_mesh(ttml.Mesh((1, self.n), ("_", "dp")))
            self.ctx.initialize_parallelism_context(ttml.autograd.DistributedConfig(enable_ddp=True, enable_tp=False))
            device = self.ctx.get_device()
            self.mapper = ttml.core.distributed.shard_tensor_to_mesh_mapper(device, 0)
            self.composer = ttml.core.distributed.concat_mesh_to_tensor_composer(device, 0)
        else:
            self.ctx.open_device()
            self.mapper = self.composer = None

    def close(self) -> None:
        ttml.close_device_mesh()  # closes the device (and fabric) in both modes

    def to_device(self, patches: np.ndarray, y: np.ndarray):
        """Host batch (from host_batch) -> (patches, labels) on device; with N devices each gets len / N samples."""
        assert len(patches) % self.n == 0, f"batch {len(patches)} not divisible by {self.n} devices"
        tt_x = ttml.autograd.Tensor.from_numpy(patches, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16, self.mapper)
        tt_y = ttml.autograd.Tensor.from_numpy(
            y.reshape(-1, 1).astype(np.uint32), ttnn.Layout.ROW_MAJOR, ttnn.DataType.UINT32, self.mapper
        )
        return tt_x, tt_y

    def to_numpy(self, t: ttml.autograd.Tensor) -> np.ndarray:
        """Device tensor -> host; batch-sharded tensors come back concatenated over all devices."""
        return t.to_numpy(composer=self.composer) if self.composer is not None else t.to_numpy()

    def sync_gradients(self, params) -> None:
        """DDP: average gradients over all devices (no-op on one device).

        "per-param" is ttml's synchronize_gradients: one all-reduce per parameter (~150 for ViT-Tiny).
        Each all-reduce is latency-bound at these sizes, so "bucketed" concatenates the gradients of
        same-shape parameters (e.g. the 12 blocks' qkv weights) along dim 0, all-reduces each bucket
        once and slices the averaged gradients back out: ~13 collectives, the same result.
        """
        if self.n == 1:
            return
        if self.grad_sync == "per-param":
            ttml.core.distributed.synchronize_gradients(params)
            return
        buckets = collections.defaultdict(list)
        for t in params.values():
            if t.is_grad_initialized():
                g = t.get_grad()
                buckets[(tuple(g.shape), g.dtype)].append((t, g))
        for (shape, _), members in buckets.items():
            grads = [g for _, g in members]
            flat = grads[0] if len(grads) == 1 else ttnn.concat(grads, dim=0)
            avg = ttml.core.distributed.synchronize_tensor(flat, [self.DP_AXIS])
            if len(members) == 1:
                members[0][0].set_grad(avg)
                continue
            k = shape[0]
            for i, (t, _) in enumerate(members):
                t.set_grad(ttnn.slice(avg, [i * k, 0, 0, 0], [(i + 1) * k, *shape[1:]]))


def host_batch(x_uint8: np.ndarray, config: ViTConfig) -> np.ndarray:
    """uint8 images -> normalized, patchified float32 model input (host side)."""
    return patchify(dataset.normalize(x_uint8), config)


def train_batches(pool, x, y, perm, global_batch, steps, config, augment, rng, lookahead):
    """Yield one epoch of (patches, labels) training batches in perm order.

    With a thread pool, up to `lookahead` upcoming batches are augmented and patchified in the
    background (numpy releases the GIL), so that host work overlaps the device step; with 8 chips the
    host prep of a 1024-image global batch (~35 ms) would otherwise stall every step. Each batch draws
    its augmentation from its own seed, so the result does not depend on the number of workers.
    """
    seeds = rng.integers(0, 2**63, size=steps)

    def make(b: int):
        idx = perm[b * global_batch : (b + 1) * global_batch]
        xb = dataset.augment(x[idx], np.random.default_rng(seeds[b])) if augment else x[idx]
        return host_batch(xb, config), y[idx]

    if pool is None:
        yield from (make(b) for b in range(steps))
        return
    pending = [pool.submit(make, b) for b in range(min(lookahead, steps))]
    for b in range(steps):
        if b + lookahead < steps:
            pending.append(pool.submit(make, b + lookahead))
        yield pending.pop(0).result()


def host_logits(devices: Devices, logits: ttml.autograd.Tensor, config: ViTConfig) -> np.ndarray:
    return devices.to_numpy(logits).astype(np.float32).reshape(-1, config.num_classes)


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
    model: ViT,
    devices: Devices,
    x: np.ndarray,
    y: np.ndarray,
    config: ViTConfig,
    batch_size: int,
    max_batches: int,
) -> tuple[float, float]:
    """Accuracy and loss on (x, y); batch_size is global. A last batch not divisible by the device
    count is padded with repeats of its first sample, and the padded logits are dropped."""
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    with no_grad():
        for i, start in enumerate(range(0, len(x), batch_size)):
            if max_batches and i >= max_batches:
                break
            xb, yb = x[start : start + batch_size], y[start : start + batch_size]
            pad = -len(xb) % devices.n
            xp = np.concatenate([xb, np.repeat(xb[:1], pad, axis=0)]) if pad else xb
            yp = np.concatenate([yb, np.repeat(yb[:1], pad, axis=0)]) if pad else yb
            tt_x, _ = devices.to_device(host_batch(xp, config), yp)
            logits = host_logits(devices, model(tt_x), config)[: len(xb)]
            correct += int((logits.argmax(axis=1) == yb).sum())
            loss_sum += host_cross_entropy(logits, yb) * len(xb)
            total += len(xb)
    model.train()
    return loss_sum / total, correct / total


def main() -> None:
    args = parse_args()
    if args.num_devices < 1:
        raise SystemExit("--num-devices must be >= 1")
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    global_batch = args.batch_size * args.num_devices

    x_train, y_train, x_test, y_test = dataset.load(args.data_dir)
    print(f"CIFAR-10: train {x_train.shape}, test {x_test.shape} from {args.data_dir}")

    devices = Devices(args.num_devices, args.grad_sync)
    ctx = devices.ctx
    pool = ThreadPoolExecutor(args.loader_workers) if args.loader_workers > 0 else None
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

        steps_per_epoch = len(x_train) // global_batch  # drop the last partial batch
        total_steps = args.max_steps or args.epochs * steps_per_epoch
        warmup_steps = int(args.warmup_epochs * steps_per_epoch)
        print(
            f"{total_steps} steps ({steps_per_epoch}/epoch), global batch {global_batch} "
            f"({args.num_devices} device(s) x {args.batch_size}), warmup {warmup_steps}"
        )

        job_id = os.environ.get("SLURM_JOB_ID")
        logger = RunLogger(
            csv_path=args.metrics,
            columns=METRIC_COLUMNS,
            wandb_project=args.wandb_project,
            run_name=args.wandb_name or (f"vit-{job_id}" if job_id else None),
            config={
                **vars(args),
                "global_batch_size": global_batch,
                **{f"model/{k}": v for k, v in dataclasses.asdict(config).items()},
            },
            tags=["vit", "cifar10"] + (["ddp", f"{args.num_devices}dev"] if args.num_devices > 1 else []),
            # x-axis: training images seen; steps are not comparable across device counts (global batch N x 128)
            step_metric="images_seen",
        )
        logger.summary(n_params=n_params, total_steps=total_steps)
        with logger:
            model.train()
            step, epoch, last_log_step = 0, 0, 0
            t0 = time.time()
            while step < total_steps:
                perm = rng.permutation(len(x_train))
                batches = train_batches(
                    pool, x_train, y_train, perm, global_batch, steps_per_epoch, config,
                    not args.no_augment, rng, 2 * args.loader_workers,
                )  # fmt: skip
                for patches, yb in batches:
                    if step >= total_steps:
                        break
                    tt_x, tt_y = devices.to_device(patches, yb)

                    lr = lr_at(step, total_steps, warmup_steps, args.lr, args.min_lr_factor)
                    optimizer.set_lr(lr)
                    optimizer.zero_grad()
                    logits = model(tt_x)
                    loss = ttml.ops.loss.cross_entropy_loss(logits, tt_y, ttml.ops.ReduceType.MEAN)
                    loss.backward(False)
                    timed = step % args.log_every == args.log_every - 1  # the step logged below
                    if timed and devices.n > 1:
                        ttnn.synchronize_device(ctx.get_device())
                        t_sync = time.time()
                    devices.sync_gradients(params)  # DDP all-reduce; clip on the averaged grads
                    if timed and devices.n > 1:
                        ttnn.synchronize_device(ctx.get_device())
                        sync_ms = (time.time() - t_sync) * 1e3
                    if args.clip_grad_norm > 0:
                        ttml.core.clip_grad_norm(params, args.clip_grad_norm, 2.0, False)
                    optimizer.step()
                    ctx.reset_graph()
                    step += 1

                    if step % args.log_every == 0 or step == 1:
                        loss_val = host_cross_entropy(host_logits(devices, logits, config), yb)
                        device_loss = float(devices.to_numpy(loss).mean())  # mean of per-device means
                        ips = (step - last_log_step) * global_batch / (time.time() - t0) if step > 1 else 0.0
                        t0, last_log_step = time.time(), step
                        print(
                            f"step {step:6d}  epoch {epoch}  lr {lr:.2e}  loss {loss_val:.4f} (device bf16 {device_loss:.4f})  {ips:7.1f} img/s"
                            + (f"  grad sync {sync_ms:.1f} ms" if devices.n > 1 and step > 1 else ""),
                            flush=True,
                        )
                        row = {"train/loss": loss_val, "train/loss_device_bf16": device_loss, "train/lr": lr}
                        if step > 1:
                            row["train/img_per_s"] = ips
                        if devices.n > 1 and step % args.log_every == 0:
                            row["train/grad_sync_ms"] = sync_ms
                        logger.log({"epoch": epoch, "images_seen": step * global_batch, **row}, step=step)

                epoch += 1
                if epoch % args.eval_every == 0 or step >= total_steps:
                    test_loss, test_acc = evaluate(
                        model, devices, x_test, y_test, config, global_batch, args.eval_batches
                    )
                    print(
                        f"== epoch {epoch}  step {step}  test loss {test_loss:.4f}  test acc {test_acc * 100:.2f}%",
                        flush=True,
                    )
                    logger.log(
                        {
                            "epoch": epoch,
                            "images_seen": step * global_batch,
                            "test/loss": test_loss,
                            "test/acc": test_acc,
                        },
                        step=step,
                    )
                    t0, last_log_step = time.time(), step
    finally:
        if pool is not None:
            pool.shutdown(cancel_futures=True)
        devices.close()


if __name__ == "__main__":
    main()
