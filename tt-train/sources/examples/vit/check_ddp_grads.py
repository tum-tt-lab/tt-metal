# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Check that N-device DDP computes the same gradients as one device (gradient accumulation).

--steps optimizer steps (AdamW, constant --lr, clip 1.0) on the 1024-image step batch train_imagenet.py
--synthetic uses (same seed, same mixup). Saves the parameters before the last step and that step's
synced gradients to an .npz. (At step 0 only the zero-initialised head has a gradient, so use >= 2.)

    python check_ddp_grads.py --num-devices 1 --out g1.npz
    python check_ddp_grads.py --num-devices 8 --grad-sync bucketed --out g8b.npz   # 1x8 mesh (sbatch sets it up)
    python check_ddp_grads.py --compare g1.npz g8b.npz

With N devices a replicated tensor comes back as N stacked copies; the copies are saved separately so
the comparison also shows whether the replicas agree.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def run(args) -> None:
    import ttml
    from ttml.common.utils import set_seed
    from ttml.models.vit import ViT, ViTConfig

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from ddp import Devices
    from train_imagenet import host_prep, micro_batches

    set_seed(args.seed)
    n = args.num_devices
    config = ViTConfig.s16_imagenet(num_blocks=args.depth)
    rng = np.random.default_rng(args.seed)
    x = rng.integers(0, 256, (args.batch_size, 3, 224, 224), np.uint8)
    y = np.arange(args.batch_size) % config.num_classes
    devices = Devices(n, args.grad_sync)
    try:
        model = ViT(config)
        params = model.parameters()
        model.train()

        def per_replica(t) -> np.ndarray:
            a = devices.to_numpy(t).astype(np.float32)
            return a.reshape(n, -1, *a.shape[1:]) if n > 1 else a[None]

        optimizer = ttml.optimizers.create_optimizer(
            {"type": "AdamW", "lr": args.lr, "beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8, "weight_decay": 0.1,
             "param_groups": [{"modules": ["bias", "gamma", "beta"], "weight_decay": 0.0}]},
            params,
        )  # fmt: skip
        global_micro = args.micro_batch_size * n
        accum = args.batch_size // global_micro
        prepped = host_prep(x, y, 0.2, [args.seed, 0], config)
        for step in range(args.steps):
            init = {k: per_replica(v) for k, v in params.items()}  # (n, ...) params before this step
            optimizer.zero_grad()
            for tt_x, tt_ya, tt_yb, _, _, lam in micro_batches(devices, prepped, global_micro):
                logits = model(tt_x)
                loss = ttml.ops.loss.cross_entropy_loss(logits, tt_ya, ttml.ops.ReduceType.MEAN)
                loss_b = ttml.ops.loss.cross_entropy_loss(logits, tt_yb, ttml.ops.ReduceType.MEAN)
                loss = loss * lam + loss_b * (1.0 - lam)
                if accum > 1:
                    loss = loss * (1.0 / accum)
                loss.backward(False)
                devices.ctx.reset_graph()
            devices.sync_gradients(params)
            if step < args.steps - 1:
                ttml.core.clip_grad_norm(params, 1.0, 2.0, False)
                optimizer.step()
        out = {}
        for k, v in params.items():
            out[f"init/{k}"] = init[k]  # (n, ...) replicas
            out[f"grad/{k}"] = per_replica(ttml.autograd.Tensor(v.get_grad(), False))  # (n, ...) replicas
        np.savez(args.out, **out)
        print(f"saved {len(params)} params' init + grads ({n} device(s), {args.grad_sync}) to {args.out}")
    finally:
        devices.close()


def compare(a_path: Path, b_path: Path) -> None:
    a, b = np.load(a_path), np.load(b_path)
    keys = sorted(k[5:] for k in a.files if k.startswith("init/"))
    print(f"{'parameter':52s} {'param rel err':>13s} {'param spread':>12s} {'grad spread':>12s} {'grad rel err':>12s}")
    worst = 0.0
    for k in keys:
        pa, pb = a[f"init/{k}"][0], b[f"init/{k}"]
        p_rel = float(np.linalg.norm(pb[0] - pa) / (np.linalg.norm(pa) + 1e-12))
        p_spread = float(np.abs(pb - pb[0]).max()) if len(pb) > 1 else 0.0
        ga, gb = a[f"grad/{k}"][0], b[f"grad/{k}"]
        spread = float(np.abs(gb - gb[0]).max()) if len(gb) > 1 else 0.0
        rel = float(np.linalg.norm(gb[0] - ga) / (np.linalg.norm(ga) + 1e-12))
        worst = max(worst, rel)
        print(f"{k[:52]:52s} {p_rel:13.3g} {p_spread:12.3g} {spread:12.3g} {rel:12.3g}")
    print(f"worst gradient relative error: {worst:.3g}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--num-devices", type=int, default=1)
    p.add_argument("--grad-sync", choices=["bucketed", "per-param"], default="bucketed")
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--micro-batch-size", type=int, default=128)
    p.add_argument("--depth", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=3, help="gradients of the last step are compared")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out", type=Path)
    p.add_argument("--compare", nargs=2, type=Path)
    args = p.parse_args()
    if args.compare:
        compare(*args.compare)
    else:
        run(args)


if __name__ == "__main__":
    main()
