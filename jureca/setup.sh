#!/bin/bash
# One-shot bootstrap. Run on a LOGIN node from anywhere inside the clone:
#   jureca/setup.sh              # init submodules, submit build, then smoke test (runs after build succeeds)
#   jureca/setup.sh --build-only
#   jureca/setup.sh --test-only
# Logs: $TT_BASE/logs/  (TT_BASE defaults to /p/project1/training2661/$USER)
set -eo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TT_BASE=${TT_BASE:-/p/project1/training2661/$USER}
LOGS=$TT_BASE/logs
mkdir -p "$LOGS"
cd "$REPO"
SB=(sbatch --parsable --chdir="$REPO" --export=ALL,TT_BASE="$TT_BASE")

case "$1" in
    --test-only)  ;;
    *)
        # Compute nodes may lack GitHub access -> fetch submodules here.
        git submodule update --init --recursive
        BUILD=$("${SB[@]}" -o "$LOGS/%x-%j.log" jureca/build.sbatch)
        echo "build job:  $BUILD   log: $LOGS/tt-build-$BUILD.log"
        [ "$1" = "--build-only" ] && exit 0
        DEP=--dependency=afterok:$BUILD ;;
esac
TEST=$("${SB[@]}" $DEP -o "$LOGS/%x-%j.log" jureca/test.sbatch)
echo "test job:   $TEST   log: $LOGS/tt-train-test-$TEST.log"
echo "watch:      squeue --me ; tail -f $LOGS/*-$TEST.log"
