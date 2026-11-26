#!/bin/bash
#SBATCH -A training2661
#SBATCH -p eval-ttbh
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -t 01:00:00
#SBATCH -J vit
# ViT on CIFAR-10 on JURECA eval-ttbh (Blackhole Galaxy). Submit from the repo root:
#   mkdir -p $TT_BASE/logs && sbatch -o $TT_BASE/logs/%x-%j.log tt-train/sources/examples/vit/train_vit.sbatch
# Smoke test:  ARGS="--max-steps 50 --eval-batches 2" sbatch -t 00:15:00 -o ... <this file>
# DDP on N chips (weak scaling: --batch-size stays per chip, global batch = N x it):
#   NDEV=4 sbatch --gres=gpu:6 -o ... <this file>
#   The N chips must form an Ethernet line; jureca/pick_line.py picks one among the chips Slurm
#   assigned, so request a couple of spare chips (they stay idle). CHIPS=a,b,c,d pins them instead.
#   A 1xN mesh graph descriptor is written to the run dir and passed via TT_MESH_GRAPH_DESC_PATH.
# wandb: offline in the job (no internet on compute nodes), upload later from a login node with
#   jureca/wandb_sync.sh.   Project: WANDB_PROJECT (default ttml-vit), off: NO_WANDB=1,
#   team: WANDB_ENTITY (default tum-tt-lab-runs, set in jureca/env.sh)
set -eo pipefail
source jureca/env.sh
NDEV=${NDEV:-1}
echo "host: $(hostname)  Slurm chips: $TT_VISIBLE_DEVICES  NDEV: $NDEV  args: ${ARGS:-<defaults>}"

RUN_DIR=$TT_BASE/runs/vit-${SLURM_JOB_ID:-local}
mkdir -p "$RUN_DIR"

if [ "$NDEV" -gt 1 ]; then
    if [ -z "$CHIPS" ]; then
        PICK=$(timeout 300 python jureca/pick_line.py "$NDEV") \
            || { echo "no $NDEV-chip Ethernet line among chips $TT_VISIBLE_DEVICES; request more (--gres)"; exit 1; }
        CHIPS=$(echo "$PICK" | sed -n 's/^CHIPS=//p')   # UMD logs to stdout too -> tagged line
    fi
    for c in ${CHIPS//,/ }; do
        [[ ",$TT_VISIBLE_DEVICES," == *",$c,"* ]] || { echo "chip $c is not in the Slurm allocation"; exit 1; }
    done
    [ "$(echo "$CHIPS" | tr ',' '\n' | grep -c .)" -eq "$NDEV" ] || { echo "CHIPS=$CHIPS is not $NDEV chips"; exit 1; }
    export TT_VISIBLE_DEVICES=$CHIPS
    export TT_MESH_GRAPH_DESC_PATH=$RUN_DIR/mgd_1x$NDEV.textproto
    cat > "$TT_MESH_GRAPH_DESC_PATH" <<EOF
mesh_descriptors {
  name: "M0"
  arch: BLACKHOLE
  device_topology { dims: [ 1, $NDEV ], dim_types: [ LINE, LINE ] }
  host_topology   { dims: [ 1, 1 ] }
  channels { count: 2 policy: STRICT }
}

top_level_instance { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
EOF
    echo "DDP over chips $TT_VISIBLE_DEVICES, MGD $TT_MESH_GRAPH_DESC_PATH"
fi

export WANDB_MODE=${WANDB_MODE:-offline} WANDB_DIR=$RUN_DIR
[ -n "$NO_WANDB" ] || WANDB_ARGS="--wandb-project ${WANDB_PROJECT:-ttml-vit}"
python -u tt-train/sources/examples/vit/train_vit.py --num-devices "$NDEV" --metrics "$RUN_DIR/metrics.csv" $WANDB_ARGS $ARGS
