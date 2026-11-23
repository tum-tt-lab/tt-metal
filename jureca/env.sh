# Environment for building/running this tt-metal clone on JURECA eval-ttbh (Blackhole Galaxy, Rocky 10).
# The compute nodes do NOT see /p/software (no Lmod) -> only node-native tools are used:
#   system clang 21, cmake+ninja from pip (in the venv), uv-managed Python 3.12,
#   hwloc headers extracted from the el10 hwloc-devel RPM into $TT_BASE/deps/sysroot.
# Usage: source jureca/env.sh   (from any directory; paths derive from this file's location)

export TT_METAL_HOME=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# Shared per-user area for caches + sysroot (keep it out of $HOME: 20 GB / 80k inode quota).
export TT_BASE=${TT_BASE:-/p/project1/training2661/$USER}

# Drop login-node module paths inherited by srun/sbatch
module --force purge >/dev/null 2>&1 || true
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v '^/p/software' | paste -sd:)
unset LD_LIBRARY_PATH CPATH LIBRARY_PATH CMAKE_PREFIX_PATH PKG_CONFIG_PATH

# Private sysroot with hwloc headers + unversioned libhwloc.so / libatomic.so (made by jureca/build.sbatch)
export TT_SYSROOT=$TT_BASE/deps/sysroot
export CMAKE_PREFIX_PATH=$TT_SYSROOT/usr
export CPATH=$TT_SYSROOT/usr/include
export LIBRARY_PATH=$TT_SYSROOT/usr/lib64

# Pure caches -> SCRATCH (JSC: no backup, files untouched for 90 days get deleted; PROJECT has a
# shared 3M-file quota). Losing them only costs a re-download / kernel recompile.
export TT_CACHE=${TT_CACHE:-/p/scratch/training2661/$USER/.cache}
export XDG_CACHE_HOME=$TT_CACHE
export UV_CACHE_DIR=$TT_CACHE/uv
export UV_LINK_MODE=copy                      # cache and venv are on different filesystems
export PIP_CACHE_DIR=$TT_CACHE/pip
export HF_HOME=$TT_CACHE/huggingface
export TORCH_HOME=$TT_CACHE/torch
export TT_METAL_CACHE=$TT_CACHE/tt-metal-cache
# Stays in PROJECT: build trees compile straight from the CPM sources, and venvs link to the uv Python
# -> a scratch purge would silently break existing builds/venvs.
export UV_PYTHON_INSTALL_DIR=$TT_BASE/.uv-python
export CPM_SOURCE_CACHE=${CPM_SOURCE_CACHE:-$TT_BASE/.cache/cpm}
export PATH=$HOME/.local/bin:$PATH            # uv lives here

# Python + runtime
export VENV_PYTHON_VERSION=3.12
export PYTHON_ENV_DIR=$TT_METAL_HOME/python_env
export PYTHONPATH=$TT_METAL_HOME
export TT_METAL_RUNTIME_ROOT=$TT_METAL_HOME
[ -f "$PYTHON_ENV_DIR/bin/activate" ] && source "$PYTHON_ENV_DIR/bin/activate"

# Slurm sets TT_VISIBLE_DEVICES from --gres=gpu:N, padded with spaces -> strip them
[ -n "$TT_VISIBLE_DEVICES" ] && export TT_VISIBLE_DEVICES=${TT_VISIBLE_DEVICES//[[:space:]]/}
cd "$TT_METAL_HOME"
