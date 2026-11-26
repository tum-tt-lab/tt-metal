#!/bin/bash
#SBATCH -A training2661
#SBATCH -p eval-ttbh
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 06:00:00
#SBATCH -J imagenet-prep
# Download ImageNet-1k (train + validation, ~155 GB Parquet) from the Hugging Face Hub and pack it for
# train_imagenet.py. Needs `hf auth login` once (token in $HF_HOME) and the dataset terms accepted at
# https://huggingface.co/datasets/ILSVRC/imagenet-1k. No chips needed; compute nodes reach the Hub.
#   sbatch -o $TT_BASE/logs/%x-%j.log tt-train/sources/examples/vit/prepare_imagenet.sbatch
# Output: $IMAGENET_ROOT/hf (Parquet) and $IMAGENET_ROOT/packed (pass as IMAGENET_DIR / --data-dir).
# The ImageNet terms forbid giving others access: everything is created owner-only (umask 077). Open it
# to teammates who accepted the terms themselves.
set -eo pipefail
source jureca/env.sh
umask 077
IMAGENET_ROOT=${IMAGENET_ROOT:-$TT_BASE/datasets/imagenet-1k}
mkdir -p "$IMAGENET_ROOT"
echo "host: $(hostname)  root: $IMAGENET_ROOT"
python -u tt-train/sources/examples/vit/prepare_imagenet.py all --hf-dir "$IMAGENET_ROOT/hf" \
    --out "$IMAGENET_ROOT/packed" --workers "${SLURM_CPUS_PER_TASK:-8}"
ls -la "$IMAGENET_ROOT/packed"
du -sh "$IMAGENET_ROOT"/*
echo "IMAGENET READY: IMAGENET_DIR=$IMAGENET_ROOT/packed"
