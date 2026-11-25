#!/bin/bash
#SBATCH -A training2661
#SBATCH -p eval-ttbh
#SBATCH -N 1
#SBATCH -c 16                 # shared box: never more than 16 for builds
#SBATCH --mem=128G
#SBATCH -t 01:30:00
#SBATCH -J tt-build
# Full (incremental on re-run) build of tt-metal + ttnn + tt-train. Submit via jureca/setup.sh,
# or from the repo root: sbatch -o $TT_BASE/logs/%x-%j.log jureca/build.sbatch
set -eo pipefail
source jureca/env.sh
set -x
cat /etc/rocky-release    # must be Rocky 10 (login nodes are Rocky 9 -> build there won't run here)

# 1) sysroot: the node has hwloc/libatomic runtime libs but not the -devel link files (no root needed)
if [ ! -f "$TT_SYSROOT/usr/include/hwloc.h" ]; then
    mkdir -p $TT_BASE/deps/rpms $TT_SYSROOT
    HWLOC_VER=$(rpm -q --qf '%{VERSION}-%{RELEASE}' hwloc-libs)
    (cd $TT_BASE/deps/rpms && dnf download hwloc-devel-$HWLOC_VER)
    (cd $TT_SYSROOT && rpm2cpio $TT_BASE/deps/rpms/hwloc-devel-$HWLOC_VER*.rpm | cpio -idm)
fi
mkdir -p $TT_SYSROOT/usr/lib64
ln -sf $(ls /usr/lib64/libhwloc.so.* | head -1) $TT_SYSROOT/usr/lib64/libhwloc.so
ln -sf /usr/lib64/libatomic.so.1                $TT_SYSROOT/usr/lib64/libatomic.so
ls $TT_SYSROOT/usr/include/hwloc.h $TT_SYSROOT/usr/lib64/libhwloc.so $TT_SYSROOT/usr/lib64/libatomic.so

# 2) venv FIRST: its Python ships headers (node has no python3-devel); cmake/ninja come from pip
# Stamp file, not bin/python: a failed create_venv.sh leaves bin/python behind -> half-empty venv
VENV_OK=$PYTHON_ENV_DIR/.jureca-venv-ok
if [ ! -f "$VENV_OK" ]; then
    rm -rf "$PYTHON_ENV_DIR"
    ./create_venv.sh --python-version 3.12 --skip-compat-check
    touch "$VENV_OK"
fi
source $PYTHON_ENV_DIR/bin/activate
uv pip install cmake ninja wandb

# 3) build
./build_metal.sh --toolchain-path $TT_METAL_HOME/jureca/clang-toolchain.cmake \
    --without-distributed --disable-profiler --build-tt-train --cpm-source-cache $CPM_SOURCE_CACHE

# 4) RHEL installs to lib64/, but tt-metal's RPATHs expect lib/
(cd build_Release && for f in lib64/*; do b=$(basename $f); [ -e lib/$b ] || ln -s ../lib64/$b lib/$b; done)

# 5) make `import ttml` work (needs both the source dir and the built _ttml module)
SP=$PYTHON_ENV_DIR/lib/python3.12/site-packages
echo "$TT_METAL_HOME/tt-train/sources/ttml"       > $SP/ttml.pth
echo "$TT_METAL_HOME/build/tt-train/sources/ttml" > $SP/_ttml.pth

# 6) checks
python -c "import ttnn, ttml; print('ttnn + ttml OK')"
ls build_Release/tt-train/sources/examples/{mnist_mlp/mnist_mlp,nano_gpt/nano_gpt}
echo "BUILD OK"
