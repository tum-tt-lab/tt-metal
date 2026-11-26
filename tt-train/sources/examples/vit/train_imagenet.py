# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""ViT-S/16 on ImageNet-1k with the "Better plain ViT baselines" recipe (Beyer et al. 2022), on one
device or data-parallel (DDP) over N devices.

Recipe (paper Appendix A; defaults of this script):
    ViT-S/16, 224px, GAP, fixed sincos2d posemb, MLP (tanh) pre-logits, zero-init head, no dropout
    batch 1024, 90 epochs, Adam(0.9, 0.999), lr 1e-3, 10k linear warmup steps, cosine to 0
    decoupled weight decay 1e-4 on weight matrices only, grad clip 1.0
    inception crop + flip + RandAugment(2, 10) + mixup(0.2)

Batch: --batch-size is the global batch per optimizer step. Each step's batch is loaded and mixed
(one mixup draw) as a whole, then cut into micro-batches of --micro-batch-size images per device,
i.e. micro x N globally; the rest is gradient accumulation: 1 chip runs 8 x 128, 8 chips 1 x (8 x 128).
Both train on identical inputs with the same steps and schedule, so 1-chip and N-chip runs differ
only in speed (and bf16 rounding).

Weight decay: big_vision subtracts wd * schedule(step) * param (not multiplied by the base lr), ttml
AdamW subtracts lr * weight_decay * param. --weight-decay 0.1 (= 1e-4 / 1e-3) is the same thing.

Metrics: every --log-every steps img/s, s/step, the share of time spent waiting for input
(perf/data_wait_frac) and, with DDP, the gradient all-reduce time; after every epoch time/epoch_min
(training wall time of that epoch, without eval and checkpointing).

--synthetic: one fixed batch kept on the device, epochs of ImageNet size (1,268,355 images) without
eval or checkpoints. Measures the device alone; use it to compare kernels and device counts.

Smoke test:
    python train_imagenet.py --data-dir /path/to/imagenet --max-steps 20 --eval-batches 2
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import ttml
from ttml.checkpointing import load_checkpoint, save_checkpoint
from ttml.common.run_logger import RunLogger
from ttml.common.utils import no_grad, set_seed
from ttml.models.vit import ViT, ViTConfig, patchify

sys.path.insert(0, str(Path(__file__).resolve().parent))
import imagenet  # noqa: E402
from ddp import Devices  # noqa: E402
from train_vit import host_cross_entropy, lr_at  # noqa: E402

METRIC_COLUMNS = [
    "epoch",
    "train/loss",
    "train/lr",
    "perf/img_per_s",
    "perf/step_s",
    "perf/data_wait_frac",
    "perf/upload_frac",
    "perf/grad_sync_ms",
    "time/epoch_min",
    "time/train_hours",
    "minival/loss",
    "minival/acc",
    "val/loss",
    "val/acc",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    m = p.add_argument_group("model (defaults: ViT-S/16 as in the paper)")
    m.add_argument("--image-size", type=int, default=224)
    m.add_argument("--patch-size", type=int, default=16)
    m.add_argument("--dim", type=int, default=384)
    m.add_argument("--heads", type=int, default=6)
    m.add_argument("--depth", type=int, default=12)
    m.add_argument("--pool", choices=["gap", "cls"], default="gap")
    m.add_argument("--posemb", choices=["sincos2d", "learned"], default="sincos2d")
    m.add_argument("--head", choices=["mlp", "linear"], default="mlp")
    k = p.add_argument_group("kernels (optimised vs unoptimised variants)")
    k.add_argument("--attention", choices=["fused", "composite"], default="fused")
    k.add_argument("--composite-layernorm", action="store_true")
    d = p.add_argument_group("devices")
    d.add_argument("--num-devices", type=int, default=1, help="> 1: data-parallel over a 1xN mesh")
    d.add_argument("--grad-sync", choices=["bucketed", "per-param"], default="bucketed")
    t = p.add_argument_group("training")
    t.add_argument("--data-dir", type=Path, default=None, help="ImageNet root with train/ and val/ class folders")
    t.add_argument("--batch-size", type=int, default=1024, help="global batch per optimizer step")
    t.add_argument("--micro-batch-size", type=int, default=128, help="per device per forward/backward")
    t.add_argument("--epochs", type=int, default=90)
    t.add_argument("--max-steps", type=int, default=0,
                   help="> 0 stops after this many optimizer steps; the LR schedule still spans --epochs")  # fmt: skip
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--min-lr-factor", type=float, default=0.0)
    t.add_argument("--warmup-steps", type=int, default=10_000)
    t.add_argument("--weight-decay", type=float, default=0.1, help="ttml/PyTorch convention, see module doc")
    t.add_argument("--clip-grad-norm", type=float, default=1.0, help="0 disables clipping")
    t.add_argument("--mixup", type=float, default=0.2, help="Beta(a, a) parameter; 0 disables")
    t.add_argument("--randaug-ops", type=int, default=2, help="0 disables RandAugment")
    t.add_argument("--randaug-magnitude", type=int, default=10)
    cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) or len(os.sched_getaffinity(0))
    t.add_argument("--workers", type=int, default=max(1, cpus - 1), help="data loader processes (default: cpus - 1)")
    t.add_argument("--prep-threads", type=int, default=4, help="host threads for mixup + patchify (steps ahead)")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--synthetic", action="store_true", help="one fixed batch kept on device: device time only")
    o = p.add_argument_group("eval / logging / checkpoints")
    o.add_argument("--eval-every", type=int, default=1, help="evaluate every N epochs")
    o.add_argument("--eval-batches", type=int, default=0, help="> 0 limits each eval to N batches")
    o.add_argument("--log-every", type=int, default=50, help="optimizer steps")
    o.add_argument("--metrics", type=Path, default=None, help="CSV file")
    o.add_argument("--checkpoint", type=Path, default=None, help="saved after each epoch, resumed from if present")
    o.add_argument("--wandb-project", default=None)
    o.add_argument("--wandb-name", default=None)
    args = p.parse_args()
    if args.num_devices < 1:
        p.error("--num-devices must be >= 1")
    global_micro = args.micro_batch_size * args.num_devices
    if args.batch_size % global_micro:
        p.error(f"--batch-size {args.batch_size} is not a multiple of micro-batch x devices = {global_micro}")
    if not args.synthetic and args.data_dir is None:
        p.error("--data-dir is required (or --synthetic)")
    return args


def mixup(x: np.ndarray, y: np.ndarray, alpha: float, rng: np.random.Generator):
    """big_vision mixup: one lam ~ Beta(a, a) per batch, lam = max(lam, 1 - lam), partner = batch rolled by 1."""
    if alpha <= 0:
        return x, y, y, 1.0
    lam = float(rng.beta(alpha, alpha))
    lam = max(lam, 1.0 - lam)
    return lam * x + (1.0 - lam) * np.roll(x, 1, axis=0), y, np.roll(y, 1), lam


def host_prep(xb: np.ndarray, yb: np.ndarray, alpha: float, seed, config: ViTConfig):
    """uint8 micro-batch -> (patches, labels a, labels b, lam), all on the host (thread-safe)."""
    x, y_a, y_b, lam = mixup(imagenet.to_float(xb), yb, alpha, np.random.default_rng(seed))
    return patchify(x, config), y_a, y_b, lam


def micro_batches(devices: Devices, prepped, size: int):
    """Host-prepped step batch -> device micro-batches of `size` images (uploaded lazily, one at a time)."""
    patches, y_a, y_b, lam = prepped
    for i in range(0, len(patches), size):
        s = slice(i, i + size)
        yield devices.images(patches[s]), devices.labels(y_a[s]), devices.labels(y_b[s]), y_a[s], y_b[s], lam


def prefetched(pool, loader, alpha: float, epoch_seed, config: ViTConfig, lookahead: int):
    """Host-prep loader batches (one per optimizer step) in `pool` up to `lookahead` ahead, in order.

    Each step has its own mixup seed, so the result does not depend on the thread count.
    """
    pending = []
    for i, (xb, yb) in enumerate(loader):
        pending.append(pool.submit(host_prep, xb.numpy(), yb.numpy(), alpha, [*epoch_seed, i], config))
        if len(pending) > lookahead:
            yield pending.pop(0).result()
    for f in pending:
        yield f.result()


def evaluate(model: ViT, devices: Devices, loader, config: ViTConfig, max_batches: int) -> tuple[float, float]:
    """Loss and accuracy; a last batch not divisible by the device count is padded and the padding dropped."""
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    with no_grad():
        for i, (xb, yb) in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            xb, yb = xb.numpy(), yb.numpy()
            pad = -len(xb) % devices.n
            xp = np.concatenate([xb, np.repeat(xb[:1], pad, axis=0)]) if pad else xb
            out = model(devices.images(patchify(imagenet.to_float(xp), config)))
            logits = devices.to_numpy(out).astype(np.float32).reshape(-1, config.num_classes)[: len(yb)]
            correct += int((logits.argmax(axis=1) == yb).sum())
            loss_sum += host_cross_entropy(logits, yb) * len(yb)
            total += len(yb)
    model.train()
    return loss_sum / total, correct / total


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    n_dev = args.num_devices
    global_micro = args.micro_batch_size * n_dev
    accum = args.batch_size // global_micro

    config = ViTConfig.s16_imagenet(
        image_size=args.image_size,
        patch_size=args.patch_size,
        embedding_dim=args.dim,
        num_heads=args.heads,
        num_blocks=args.depth,
        pool=args.pool,
        posemb=args.posemb,
        head=args.head,
        attention=args.attention,
        use_composite_layernorm=args.composite_layernorm,
    )

    pool = None
    if args.synthetic:
        n_train = imagenet.TRAIN_SPLIT_SIZE
        rng0 = np.random.default_rng(args.seed)
        fake_x = rng0.integers(0, 256, (args.batch_size, 3, args.image_size, args.image_size), np.uint8)
        fake_y = np.arange(args.batch_size) % config.num_classes
    else:
        ds = imagenet.build_datasets(args.data_dir, args.image_size, args.randaug_ops, args.randaug_magnitude)
        n_train = len(ds["train"])
        sampler = imagenet.EpochShuffle(n_train, args.seed)
        train_loader = imagenet.make_loader(ds["train"], args.batch_size, args.workers, sampler)
        eval_workers = min(args.workers, 4)
        eval_loaders = {name: imagenet.make_loader(ds[name], global_micro, eval_workers) for name in ("minival", "val")}
        pool = ThreadPoolExecutor(args.prep_threads)
        # Fork the (persistent) loader workers now, while this process is small: once the devices are open
        # it maps a lot of device memory, and fork() then fails with ENOMEM (seen with an 8-chip mesh).
        # The batches these first iterators prefetch are discarded when each loader is iterated for real.
        for loader in (train_loader, *eval_loaders.values()):
            iter(loader)
        print(f"ImageNet: train {n_train}, minival {len(ds['minival'])}, val {len(ds['val'])}, workers {args.workers}")

    steps_per_epoch = n_train // args.batch_size
    total_steps = args.epochs * steps_per_epoch  # LR schedule length
    stop_step = min(args.max_steps, total_steps) if args.max_steps else total_steps
    epochs = -(-stop_step // steps_per_epoch)

    devices = Devices(n_dev, args.grad_sync)
    ctx = devices.ctx
    try:
        model = ViT(config)
        params = model.parameters()
        n_params = sum(int(np.prod(t.shape())) for t in params.values())
        print(
            f"ViT: {config}\n  tokens {config.seq_len} (padded {config.padded_seq_len}), params {n_params / 1e6:.2f}M\n"
            f"  schedule {total_steps} steps ({steps_per_epoch}/epoch), stop at {stop_step}, batch {args.batch_size} = {accum} accum x "
            f"{n_dev} device(s) x {args.micro_batch_size}"
        )

        no_decay = ["bias", "gamma", "beta"] + (["pos_emb"] if config.posemb == "learned" else [])
        optimizer = ttml.optimizers.create_optimizer(
            {
                "type": "AdamW",
                "lr": args.lr,
                "beta1": 0.9,
                "beta2": 0.999,
                "epsilon": 1e-8,
                "weight_decay": args.weight_decay,
                "param_groups": [{"modules": no_decay, "weight_decay": 0.0}],
            },
            params,
        )

        start_epoch, step, train_time = 0, 0, 0.0
        if args.checkpoint and args.checkpoint.exists():
            header = load_checkpoint(str(args.checkpoint), model_params=params, optimizer=optimizer)
            start_epoch, step, train_time = header["epoch"], header["step"], header["train_time_s"]
            print(f"resumed from {args.checkpoint}: epoch {start_epoch}, step {step}")

        job_id = os.environ.get("SLURM_JOB_ID")
        logger = RunLogger(
            csv_path=args.metrics,
            columns=METRIC_COLUMNS,
            wandb_project=args.wandb_project,
            run_name=args.wandb_name or (f"vits16-{n_dev}dev-{job_id}" if job_id else None),
            config={
                **vars(args),
                "accum_steps": accum,
                "global_micro_batch": global_micro,
                "n_params": n_params,
                "total_steps": total_steps,
                "stop_step": stop_step,
                "steps_per_epoch": steps_per_epoch,
                **{f"model/{k}": v for k, v in dataclasses.asdict(config).items()},
            },
            tags=["vit", "imagenet", "vits16", f"{n_dev}dev"]
            + (["ddp"] if n_dev > 1 else [])
            + (["synthetic"] if args.synthetic else []),
        )

        with logger:
            model.train()
            for epoch in range(start_epoch, epochs):
                if step >= stop_step:
                    break
                epoch_seed = [args.seed, epoch]  # a resumed run sees the same order and mixup draws
                if args.synthetic:
                    # One step batch, uploaded once and reused: no host work inside the timed loop.
                    fixed = list(micro_batches(devices, host_prep(fake_x, fake_y, args.mixup, epoch_seed, config), global_micro))
                    steps = iter(lambda: iter(fixed), None)
                else:
                    sampler.set_epoch(epoch)
                    prepped = prefetched(pool, train_loader, args.mixup, epoch_seed, config, args.prep_threads)
                    steps = (micro_batches(devices, b, global_micro) for b in prepped)

                epoch_start = window_start = time.time()
                window_steps, data_wait, upload, sync_ms = 0, 0.0, 0.0, None
                for _ in range(steps_per_epoch):
                    if step >= stop_step:
                        break
                    lr = lr_at(step, total_steps, args.warmup_steps, args.lr, args.min_lr_factor)
                    optimizer.set_lr(lr)
                    optimizer.zero_grad()
                    log_now = (step + 1) % args.log_every == 0 or step == 0
                    t = time.time()
                    micros = next(steps)
                    data_wait += time.time() - t
                    for micro in range(accum):
                        t = time.time()
                        tt_x, tt_ya, tt_yb, y_a, y_b, lam = next(micros)  # host -> device copy
                        upload += time.time() - t
                        logits = model(tt_x)
                        loss = ttml.ops.loss.cross_entropy_loss(logits, tt_ya, ttml.ops.ReduceType.MEAN)
                        if lam < 1.0:
                            loss_b = ttml.ops.loss.cross_entropy_loss(logits, tt_yb, ttml.ops.ReduceType.MEAN)
                            loss = loss * lam + loss_b * (1.0 - lam)
                        if accum > 1:
                            loss = loss * (1.0 / accum)
                        loss.backward(False)
                        if log_now and micro == accum - 1:
                            # Last micro-batch only (all devices); reading it also syncs the device.
                            z = devices.to_numpy(logits).astype(np.float32).reshape(-1, config.num_classes)
                            loss_val = lam * host_cross_entropy(z, y_a) + (1 - lam) * host_cross_entropy(z, y_b)
                        ctx.reset_graph()
                    if log_now and n_dev > 1:
                        devices.synchronize()
                        t_sync = time.time()
                    devices.sync_gradients(params)  # DDP all-reduce; clip on the averaged gradients
                    if log_now and n_dev > 1:
                        devices.synchronize()
                        sync_ms = (time.time() - t_sync) * 1e3
                    if args.clip_grad_norm > 0:
                        ttml.core.clip_grad_norm(params, args.clip_grad_norm, 2.0, False)
                    optimizer.step()
                    step += 1
                    window_steps += 1

                    if log_now:
                        devices.synchronize()
                        elapsed = time.time() - window_start
                        ips = window_steps * args.batch_size / elapsed
                        row = {
                            "epoch": epoch,
                            "train/loss": loss_val,
                            "train/lr": lr,
                            "perf/img_per_s": ips,
                            "perf/step_s": elapsed / window_steps,
                            "perf/data_wait_frac": data_wait / elapsed,  # blocked on the input pipeline
                            "perf/upload_frac": upload / elapsed,  # host -> device copies (overlap device work)
                            "time/train_hours": (train_time + time.time() - epoch_start) / 3600,
                        }
                        if sync_ms is not None:
                            row["perf/grad_sync_ms"] = sync_ms
                        eta_h = (stop_step - step) * row["perf/step_s"] / 3600
                        print(
                            f"step {step:7d}  ep {epoch}  lr {lr:.2e}  loss {loss_val:.4f}  {ips:7.1f} img/s  "
                            f"{row['perf/step_s']:.3f} s/step  data-wait {row['perf/data_wait_frac']:.0%}  upload {row['perf/upload_frac']:.0%}"
                            + (f"  grad sync {sync_ms:.1f} ms" if sync_ms is not None else "")
                            + f"  eta {eta_h:.1f}h",
                            flush=True,
                        )
                        logger.log(row, step=step)
                        window_start, window_steps, data_wait, upload = time.time(), 0, 0.0, 0.0

                devices.synchronize()
                epoch_s = time.time() - epoch_start
                train_time += epoch_s
                row = {"epoch": epoch, "time/epoch_min": epoch_s / 60, "time/train_hours": train_time / 3600}
                if not args.synthetic and ((epoch + 1) % args.eval_every == 0 or step >= stop_step):
                    for name, loader in eval_loaders.items():
                        row[f"{name}/loss"], row[f"{name}/acc"] = evaluate(model, devices, loader, config, args.eval_batches)
                print(
                    f"== epoch {epoch}  step {step}  epoch time {epoch_s / 60:.2f} min"
                    + (f"  minival {row['minival/acc']:.2%}  val {row['val/acc']:.2%}" if "val/acc" in row else ""),
                    flush=True,
                )
                logger.log(row, step=step)
                if args.checkpoint and not args.synthetic:
                    save_checkpoint(
                        str(args.checkpoint),
                        header={"epoch": epoch + 1, "step": step, "train_time_s": train_time},
                        model_params=params,
                        optimizer=optimizer,
                    )
    finally:
        if pool is not None:
            pool.shutdown(cancel_futures=True)
        devices.close()


if __name__ == "__main__":
    main()
