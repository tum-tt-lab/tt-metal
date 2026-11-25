#!/bin/bash
#SBATCH -A training2661
#SBATCH -p eval-ttbh
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -t 01:00:00
#SBATCH -J vit
# ViT on CIFAR-10, one Blackhole chip on JURECA. Submit from the repo root:
#   mkdir -p $TT_BASE/logs && sbatch -o $TT_BASE/logs/%x-%j.log tt-train/sources/examples/vit/train_vit.sbatch
# Smoke test:  ARGS="--max-steps 50 --eval-batches 2" sbatch -t 00:15:00 -o ... <this file>
# wandb: offline in the job (no internet on compute nodes), upload later from a login node with
#   jureca/wandb_sync.sh.   Project: WANDB_PROJECT (default ttml-vit), team: WANDB_ENTITY, off: NO_WANDB=1
set -eo pipefail
source jureca/env.sh
echo "host: $(hostname)  chip: $TT_VISIBLE_DEVICES  args: ${ARGS:-<defaults>}"

RUN_DIR=$TT_BASE/runs/vit-${SLURM_JOB_ID:-local}
mkdir -p "$RUN_DIR"
export WANDB_MODE=${WANDB_MODE:-offline} WANDB_DIR=$RUN_DIR
[ -n "$NO_WANDB" ] || WANDB_ARGS="--wandb-project ${WANDB_PROJECT:-ttml-vit}"
python -u tt-train/sources/examples/vit/train_vit.py --metrics "$RUN_DIR/metrics.csv" $WANDB_ARGS $ARGS
