#!/bin/bash
#SBATCH -A training2661
#SBATCH -p eval-ttbh
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -t 00:15:00
#SBATCH -J tt-train-test
# Smoke test for tt-train on ONE Blackhole chip:
#   1) build artifacts exist   2) `import ttnn, ttml`
#   3) tiny ttml forward/backward on device   4) one tt-train pytest (gelu)
# Submit via jureca/setup.sh --test-only, or from the repo root:
#   sbatch -o $TT_BASE/logs/%x-%j.log jureca/test.sbatch
set -eo pipefail
source jureca/env.sh
echo "host: $(hostname)  chip: $TT_VISIBLE_DEVICES"

echo "=== [1/4] build artifacts"
[ -x "$PYTHON_ENV_DIR/bin/python" ] || { echo "FAIL: no venv at $PYTHON_ENV_DIR (run jureca/setup.sh)"; exit 1; }
ls build_Release/tt-train/sources/examples/mnist_mlp/mnist_mlp \
   || { echo "FAIL: tt-train not built (run jureca/setup.sh)"; exit 1; }

echo "=== [2/4] python import"
timeout 120 python -c "import ttnn, ttml; print('ttnn + ttml OK:', ttml.__file__)"

echo "=== [3/4] tiny autograd step on device"
timeout 300 python - <<'PY'
import numpy as np, ttnn, ttml
ctx = ttml.autograd.AutoContext.get_instance()
ctx.open_device()
try:
    x = ttml.autograd.Tensor.from_numpy(np.random.randn(1, 1, 32, 32).astype(np.float32), layout=ttnn.Layout.TILE)
    x.set_requires_grad(True)   # from_numpy defaults to False -> no backward node
    y = ttml.ops.unary.relu(x)
    y.backward(False)           # seeds dL/dy = ones
    xn = x.to_numpy()
    err = float(np.abs(y.to_numpy() - np.maximum(xn, 0)).max())
    gerr = float(np.abs(ttnn.to_torch(x.get_grad()).float().numpy() - (xn > 0)).max())
    print(f"relu max abs err vs numpy: fwd {err:.4g}  grad {gerr:.4g}")
    assert err < 1e-2 and gerr < 1e-2, "relu mismatch"
    print("device autograd OK")
finally:
    ctx.close_device()
PY

echo "=== [4/4] pytest tt-train/tests/python/test_gelu.py"
cd $TT_METAL_HOME/tt-train
timeout 600 python -m pytest -q --tb=native tests/python/test_gelu.py   # --tb=native: default tb hangs on stuck chips

echo "ALL TT-TRAIN SMOKE TESTS PASSED"
