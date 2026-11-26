#!/bin/bash
#SBATCH -A training2661
#SBATCH -p eval-ttbh
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -t 00:20:00
#SBATCH -J ddp-grads
# One-step gradient check of DDP against one device (see check_ddp_grads.py). From the repo root:
#   sbatch -o $TT_BASE/logs/%x-%j.log tt-train/sources/examples/vit/check_ddp_grads.sbatch            # 1 chip
#   NDEV=8 sbatch --gres=gpu:16 -o ... tt-train/sources/examples/vit/check_ddp_grads.sbatch            # both syncs
# Writes $TT_BASE/runs/ddp-grads/g<NDEV>[-<sync>].npz; compare with check_ddp_grads.py --compare.
set -eo pipefail
source jureca/env.sh
NDEV=${NDEV:-1}
RUN_DIR=$TT_BASE/runs/ddp-grads
mkdir -p "$RUN_DIR"
source jureca/mesh_setup.sh
S=tt-train/sources/examples/vit/check_ddp_grads.py
if [ "$NDEV" -eq 1 ]; then
    timeout 900 python -u $S --out "$RUN_DIR/g1.npz"
else
    for sync in bucketed per-param; do
        timeout 900 python -u $S --num-devices "$NDEV" --grad-sync $sync --out "$RUN_DIR/g$NDEV-$sync.npz"
    done
fi
