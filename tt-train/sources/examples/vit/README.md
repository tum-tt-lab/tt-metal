# ViT on CIFAR-10 (Python, ttml)

Trains a Vision Transformer from scratch on CIFAR-10 with ttml autograd on one device.

| File | What |
|---|---|
| `ttml/models/vit/` (in `sources/ttml/`) | `ViT`, `ViTConfig`, `patchify`, `select_first_token` |
| `train_vit.py` | training loop: AdamW, warmup + cosine LR, grad clipping, test eval each epoch |
| `dataset.py` | CIFAR-10 binary loader, crop/flip augmentation; run it directly to download |
| `train_vit.sbatch` | JURECA eval-ttbh job (1 chip) |

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
ARGS="--max-steps 50 --eval-batches 2" TEST=1 sbatch -t 00:20:00 -o $TT_BASE/logs/%x-%j.log \
    tt-train/sources/examples/vit/train_vit.sbatch          # tests + smoke run
sbatch -t 04:00:00 -o $TT_BASE/logs/%x-%j.log tt-train/sources/examples/vit/train_vit.sbatch
```

Metrics go to `$TT_BASE/runs/vit-<jobid>/metrics.csv`. `--help` lists the model and training flags.
