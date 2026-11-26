# ViT with ttml: CIFAR-10 and ImageNet-1k (Python)

Trains a Vision Transformer from scratch with ttml autograd on one device.

| File | What |
|---|---|
| `ttml/models/vit/` (in `sources/ttml/`) | `ViT`, `ViTConfig` (+ `ViTConfig.s16_imagenet()`), `patchify`, autograd ops |
| `train_vit.py`, `dataset.py`, `train_vit.sbatch` | CIFAR-10: ViT-Tiny, 4x4 patches, 32px |
| `train_imagenet.py`, `imagenet.py`, `train_imagenet.sbatch` | ImageNet-1k: ViT-S/16 baseline of [Beyer et al. 2022](https://arxiv.org/abs/2205.01580) |

## Model

Details in the module docstring.

- **Patch embedding**: host-side `patchify` + `LinearLayer(C*P*P -> dim)`, the same as a stride-P conv.
- **Padding**: tokens are padded to a multiple of 32 (tile height); a key mask hides the padding in SDPA.
- **Encoder**: pre-LN blocks (MHA + 4x GELU MLP), non-causal mask.
- **Options**: `pool` `cls`/`gap`, `posemb` `learned`/`sincos2d`, `head` `linear`/`mlp` (tanh pre-logits),
  `attention` `fused`/`composite`, `use_composite_layernorm`.

| | CIFAR-10 (`ViTConfig()`) | ImageNet (`ViTConfig.s16_imagenet()`) |
|---|---|---|
| size | Tiny: dim 192, 12 blocks, 3 heads, 5.4M params | S/16: dim 384, 12 blocks, 6 heads, ~22M params |
| tokens | [CLS] + 64 patches, padded to 96 | 196 patches, padded to 224 |
| pool / posemb / head | [CLS] / learned / linear | GAP / fixed sincos2d / MLP (tanh) |
| dropout | 0.1 | none |

## CIFAR-10

```bash
source jureca/env.sh && python tt-train/sources/examples/vit/dataset.py   # once, login node (needs internet)
ARGS="--max-steps 50 --eval-batches 2" sbatch -t 00:15:00 -o $TT_BASE/logs/%x-%j.log \
    tt-train/sources/examples/vit/train_vit.sbatch          # smoke run
sbatch -t 04:00:00 -o $TT_BASE/logs/%x-%j.log tt-train/sources/examples/vit/train_vit.sbatch
```

`source jureca/env.sh` first, so the data lands in `$TT_CACHE/datasets/`, where the job looks for it.

## ImageNet-1k: ViT-S/16 baseline

`IMAGENET_DIR` must contain `train/` and `val/` with one folder per WordNet id (the official val set
sorted into class folders). The minival split (1% of train) is chosen inside the script.

```bash
export IMAGENET_DIR=/path/to/imagenet
ARGS="--max-steps 20 --eval-batches 2" RUN_NAME=smoke sbatch -t 00:30:00 -o $TT_BASE/logs/%x-%j.log \
    tt-train/sources/examples/vit/train_imagenet.sbatch     # smoke run
RUN_NAME=vits16-base sbatch -o $TT_BASE/logs/%x-%j.log tt-train/sources/examples/vit/train_imagenet.sbatch
# continue after the job's time limit (resumes from $TT_BASE/runs/vits16-base/ckpt.bin):
RUN_NAME=vits16-base sbatch --dependency=afterany:<job id> -o $TT_BASE/logs/%x-%j.log \
    tt-train/sources/examples/vit/train_imagenet.sbatch
# DDP on 8 chips (ask for spare chips; jureca/pick_line.py picks 8 Ethernet-linked ones):
NDEV=8 RUN_NAME=vits16-8dev sbatch --gres=gpu:12 -o $TT_BASE/logs/%x-%j.log \
    tt-train/sources/examples/vit/train_imagenet.sbatch
```

`--batch-size` (1024) is the global batch per optimizer step on any number of chips; each chip runs
`--micro-batch-size` (128) images per forward/backward and the rest is gradient accumulation (1 chip:
8 x 128, 8 chips: 1 x 8 x 128). So a 1-chip and an 8-chip run take the same steps with the same
schedule and differ only in time. DDP (`ddp.py`) is the vit-ddp branch's: replicated parameters, batch
sharded over a 1xN mesh, bucketed gradient all-reduce.

Recipe = the paper's Appendix A config: batch 1024 (8 x 128 with gradient accumulation), 90 epochs,
Adam (0.9, 0.999), lr 1e-3, 10k warmup steps, cosine to 0, weight decay on weight matrices only,
clip 1.0, inception crop (5-100% area) + flip + RandAugment(2, 10) + mixup(0.2), eval resize 256 / crop 224.

Known differences from big_vision:

- weight decay is `--weight-decay 0.1` because ttml AdamW multiplies it by lr (big_vision `wd=1e-4` is not)
- torchvision RandAugment, not big_vision's TF op set (same strength scale: 10 = maximum)
- ttml initialisers (normal 0.02) instead of big_vision's (lecun normal / xavier uniform)
- bf16 activations and (ttml default) parameters; big_vision keeps fp32 params with a bf16 Adam first moment
- tokens padded 196 -> 224, i.e. ~14% extra token compute (masked, so the math is unchanged)

Metrics per run in `$TT_BASE/runs/$RUN_NAME/metrics-<job>.csv` and wandb (team `tum-tt-lab-runs`,
project `ttml-vit`, run `<RUN_NAME>-<job id>`, grouped by `RUN_NAME`). The compute nodes reach
api.wandb.ai, so after a one-time `wandb login` on a login node (key in `~/.netrc`) runs log **live**.
Without a key they log offline into the run dir; upload them with `jureca/wandb_sync.sh`.

## Measuring speed-ups

Each log line has `perf/img_per_s`, `perf/step_s`, `perf/data_wait_frac` (share of wall time spent
waiting for the input pipeline, including host-side mixup/patchify/upload), with DDP
`perf/grad_sync_ms`, and `time/train_hours` (training wall time without eval and checkpointing; with
`minival/acc` this gives time-to-accuracy). After every epoch: `time/epoch_min`.

- **Per-epoch comparison** (1 vs 8 chips, kernel variants): `time/epoch_min`. `--synthetic --epochs 1`
  runs one ImageNet-sized epoch (1,268,355 images, 1238 steps) of device time only, no dataset needed.

- **Kernel comparisons** (fused vs composite attention/LayerNorm, ...): use `--synthetic`. It keeps one
  batch on the device, so `perf/step_s` is device time only. The math is unchanged, so step time decides
  everything: speed-up = baseline `step_s` / optimised `step_s` after warm-up (skip the first log lines,
  they include kernel compilation).
- **Changes that touch numerics** (lower precision, approximations): also compare `time/train_hours`
  to a fixed `minival/acc`, because fewer seconds per step do not help if more steps are needed.
- **Real runs**: if `perf/data_wait_frac` is well above 0, the input pipeline (CPU) is the bottleneck
  and device optimisations will not show up in epoch time.
