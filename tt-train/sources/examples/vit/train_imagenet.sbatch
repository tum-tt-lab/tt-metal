#!/bin/bash
#SBATCH -A training2661
#SBATCH -p eval-ttbh
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00
#SBATCH -J vits16
# ViT-S/16 on ImageNet-1k (Beyer et al. 2022 recipe) on JURECA eval-ttbh. Submit from the repo root:
#   IMAGENET_DIR=/path/to/imagenet RUN_NAME=vits16-base sbatch -o $TT_BASE/logs/%x-%j.log \
#       tt-train/sources/examples/vit/train_imagenet.sbatch
# IMAGENET_DIR holds train/ and val/ class folders. The run lives in $TT_BASE/runs/$RUN_NAME and resumes
# from its checkpoint (saved every epoch), so a 90-epoch run is a chain of jobs with the same RUN_NAME:
#   sbatch --dependency=afterany:<previous job id> ... (same command)
# Smoke test:   ARGS="--max-steps 20 --eval-batches 2" RUN_NAME=smoke sbatch -t 00:30:00 ...
# Device only:  ARGS="--synthetic --epochs 1" RUN_NAME=synth-1dev sbatch -t 01:00:00 ...
# DDP on N chips (global batch stays --batch-size; each chip gets --micro-batch-size per forward):
#   NDEV=8 sbatch --gres=gpu:12 ...   (request spare chips: jureca/pick_line.py picks N Ethernet-linked
#   ones among those Slurm assigned; CHIPS=a,b,... pins them instead. A 1xN mesh graph descriptor is
#   written to the run dir and passed via TT_MESH_GRAPH_DESC_PATH.)
# wandb (team WANDB_ENTITY from jureca/env.sh, project WANDB_PROJECT, default ttml-vit; NO_WANDB=1: off):
#   compute nodes reach the wandb server (wandb 0.31 default: api.forge.coreweave.com), so the run logs live (online) when an API key is present
#   (`wandb login` once on a login node writes ~/.netrc, which jobs see). Without a key it logs offline
#   into the run dir; upload later with jureca/wandb_sync.sh. WANDB_MODE=offline forces offline.
set -eo pipefail
source jureca/env.sh
NDEV=${NDEV:-1}
RUN_NAME=${RUN_NAME:-vits16-${NDEV}dev-${SLURM_JOB_ID:-local}}
echo "host: $(hostname)  Slurm chips: $TT_VISIBLE_DEVICES  NDEV: $NDEV  cpus: $SLURM_CPUS_PER_TASK  args: ${ARGS:-<defaults>}"

RUN_DIR=$TT_BASE/runs/$RUN_NAME
mkdir -p "$RUN_DIR"

source jureca/mesh_setup.sh   # NDEV > 1: pick an Ethernet line of chips, write the mesh descriptor

if [ -z "$WANDB_MODE" ]; then
    if [ -n "$WANDB_API_KEY" ] || grep -qsE "api\.wandb\.ai|forge\.coreweave\.com" ~/.netrc; then WANDB_MODE=online; else WANDB_MODE=offline; fi
fi
export WANDB_MODE WANDB_DIR=$RUN_DIR WANDB_RUN_GROUP=$RUN_NAME
echo "wandb: $WANDB_MODE"
[ -n "$NO_WANDB" ] || WANDB_ARGS="--wandb-project ${WANDB_PROJECT:-ttml-vit} --wandb-name $RUN_NAME-${SLURM_JOB_ID:-local}"
[[ " $ARGS " == *" --synthetic "* ]] || DATA_ARGS="--data-dir ${IMAGENET_DIR:?set IMAGENET_DIR} --checkpoint $RUN_DIR/ckpt.bin"
# One CSV per job (a resumed run must not overwrite the previous job's rows).
python -u tt-train/sources/examples/vit/train_imagenet.py --num-devices "$NDEV" \
    --metrics "$RUN_DIR/metrics-${SLURM_JOB_ID:-local}.csv" $DATA_ARGS $WANDB_ARGS $ARGS
