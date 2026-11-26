# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""One device or data-parallel (DDP) over a 1xN mesh, for the ViT examples.

Taken from the `Devices` class of train_vit.py on the vit-ddp branch (Jonas Senft-Raiss), so both
examples shard batches and average gradients the same way.

N devices: a 1xN mesh with DDP. Batch tensors are sharded along dim 0 (mapper) and outputs are
concatenated back along dim 0 (composer). Tensors created without a mapper (parameters, buffers such
as the attention mask) are replicated on every device. Needs TT_MESH_GRAPH_DESC_PATH pointing at a 1xN
mesh graph descriptor whose chips are Ethernet-linked (train_imagenet.sbatch with NDEV=N sets this up).
"""

from __future__ import annotations

import collections

import numpy as np
import ttnn
import ttml


class Devices:
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

    def synchronize(self) -> None:
        ttnn.synchronize_device(self.ctx.get_device())

    def images(self, patches: np.ndarray) -> ttml.autograd.Tensor:
        """Host (B, 1, S, D) model input -> device; with N devices each gets B / N samples."""
        assert len(patches) % self.n == 0, f"batch {len(patches)} not divisible by {self.n} devices"
        return ttml.autograd.Tensor.from_numpy(patches, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16, self.mapper)

    def labels(self, y: np.ndarray) -> ttml.autograd.Tensor:
        return ttml.autograd.Tensor.from_numpy(
            y.reshape(-1, 1).astype(np.uint32), ttnn.Layout.ROW_MAJOR, ttnn.DataType.UINT32, self.mapper
        )

    def to_numpy(self, t: ttml.autograd.Tensor) -> np.ndarray:
        """Device tensor -> host; batch-sharded tensors come back concatenated over all devices."""
        return t.to_numpy(composer=self.composer) if self.composer is not None else t.to_numpy()

    def sync_gradients(self, params) -> None:
        """DDP: average gradients over all devices (no-op on one device).

        "per-param" is ttml's synchronize_gradients: one all-reduce per parameter. Each all-reduce is
        latency-bound at these sizes, so "bucketed" concatenates the gradients of same-shape parameters
        (e.g. the 12 blocks' qkv weights) along dim 0, all-reduces each bucket once and slices the
        averaged gradients back out: the same result with ~13 collectives instead of ~150.
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
