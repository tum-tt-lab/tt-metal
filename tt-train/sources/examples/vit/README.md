# ViT on CIFAR-10 (Python, ttml)

Trains a Vision Transformer from scratch on CIFAR-10 with ttml autograd on one device, or data-parallel
(DDP) on several chips of the Blackhole Galaxy.

| File | What |
|---|---|
| `ttml/models/vit/` (in `sources/ttml/`) | `ViT`, `ViTConfig`, `patchify`, `select_first_token` |
| `train_vit.py` | training loop: AdamW, warmup + cosine LR, grad clipping, test eval each epoch, DDP |
| `dataset.py` | CIFAR-10 binary loader, crop/flip augmentation; run it directly to download |
| `train_vit.sbatch` | JURECA eval-ttbh job (1 chip, or `NDEV=N` chips); logs to CSV and offline wandb |
| `jureca/pick_line.py` (repo root) | picks N Ethernet-linked chips among the Slurm-assigned ones |

## Model

Defaults are ViT-Tiny (dim 192, 12 blocks, 3 heads, MLP 768, ~5.4M params) on native 32x32 images
with 4x4 patches. Details (see the module docstring):

- **Patch embedding**: host-side `patchify` + `LinearLayer(48 -> dim)`, the same as a stride-4 conv.
- **[CLS]**: input row 0 is zero, so after the patch Linear it is `bias + pos[0]`, a learnable vector.
- **Padding**: 65 tokens are padded to 96 (multiple of 32); a key mask hides the padding in SDPA.
- **Encoder**: NanoGPT's pre-LN `GPTBlock`, non-causal mask. **Head**: token 0 -> LayerNorm -> Linear (zero-init).

## Run

```bash
python tt-train/sources/examples/vit/dataset.py      # once, on a login node (compute nodes have no internet)
# JURECA, from the repo root:
ARGS="--max-steps 50 --eval-batches 2" sbatch -t 00:15:00 -o $TT_BASE/logs/%x-%j.log \
    tt-train/sources/examples/vit/train_vit.sbatch          # smoke run
sbatch -t 04:00:00 -o $TT_BASE/logs/%x-%j.log tt-train/sources/examples/vit/train_vit.sbatch
NDEV=8 sbatch --gres=gpu:12 -t 00:20:00 -o $TT_BASE/logs/%x-%j.log \
    tt-train/sources/examples/vit/train_vit.sbatch          # DDP on 8 chips
```

Metrics go to `$TT_BASE/runs/vit-<jobid>/metrics.csv` and to an offline wandb run in the same
directory (team `tum-tt-lab-runs`, project `ttml-vit`; set `WANDB_PROJECT` / `WANDB_ENTITY` to change it, `NO_WANDB=1` to
turn it off). Upload with `jureca/wandb_sync.sh` from a login node. `--help` lists the model and
training flags.

## Data parallel (DDP)

`--num-devices N` (`NDEV=N` in the sbatch) opens a 1xN mesh with the DDP parallelism context:

- Parameters (and the attention mask) are replicated; every device starts from the same weights.
- `--batch-size` is **per device**; each step takes a global batch of `N * batch-size` images and
  shards it along the batch dim (`shard_tensor_to_mesh_mapper(device, 0)`). Logits/losses come back
  with `concat_mesh_to_tensor_composer(device, 0)` for logging and eval.
- After backward, the gradients are averaged over the mesh (all-reduce over the Ethernet fabric), then
  clipped and applied by AdamW, so all replicas stay identical. `--grad-sync bucketed` (default)
  concatenates same-shape gradients (the 12 blocks' qkv weights, ...) and all-reduces each group
  once (~13 collectives); `per-param` is ttml's `synchronize_gradients` (one per parameter, ~150).
  Both give bit-identical losses; bucketed is a bit faster.
- Host data prep (augment + patchify) runs in `--loader-workers` threads ahead of the device; at 8
  chips the 1024-image global batch would otherwise make the host the bottleneck.
- Learning rate and warmup are **not** rescaled automatically: the global batch grows with N, so
  steps per epoch shrink by N (warmup is in epochs, so it shrinks too). Tune `--lr` for large N.

The fabric needs a mesh graph descriptor matching the chips: the sbatch lets `jureca/pick_line.py`
choose N chips that form an Ethernet line among the Slurm-assigned ones, writes a 1xN descriptor to the
run dir and sets `TT_MESH_GRAPH_DESC_PATH`. Ask Slurm for a few spare chips (`--gres=gpu:6` for 4,
`gpu:12` for 8; consecutive chip ids are not always linked), or pin them with `CHIPS=a,b,...`.

Throughput (ViT-Tiny, 128 images per chip, JURECA jrc0950, 2026-11-26):

| chips | img/s | speedup | grad all-reduce |
|---|---|---|---|
| 1 | 2530 | 1.0x | - |
| 2 | 4850 | 1.9x | ~10 ms |
| 4 | 9690 | 3.8x | ~10 ms |
| 8 | 17900 | 7.1x | ~13 ms |

Full run, default hyperparameters (lr 1e-3, 100 epochs): 8 chips (global batch 1024) reach **82.6%**
test accuracy in 5 min 40 s, vs 81.1% in 36 min on 1 chip (batch 128).

If the job exits with "no N-chip Ethernet line", the assigned chips are not linked that way (e.g. chips
12-23 contained no 8-chip line); resubmit with more chips in `--gres` or pin known-good `CHIPS`
(chips 0-7 worked as the line 0,1,2,3,7,6,5,4).
