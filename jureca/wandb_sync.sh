#!/bin/bash
# Upload offline wandb runs from compute-node jobs. Run on a LOGIN node (compute nodes have no internet).
#   jureca/wandb_sync.sh                    # every unsynced run under $TT_BASE/runs/*/wandb/
#   jureca/wandb_sync.sh <run dir>...       # specific offline-run-* directories
# Needs a one-time `wandb login` (stores the API key in ~/.netrc). Already-synced runs are skipped.
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
if [ $# -eq 0 ]; then
    shopt -s nullglob
    set -- "$TT_BASE"/runs/*/wandb/offline-run-*
    [ $# -gt 0 ] || { echo "no offline runs under $TT_BASE/runs/*/wandb/"; exit 0; }
fi
wandb sync --yes "$@"
