# tt-metal + tt-train on JURECA (eval-ttbh, Blackhole Galaxy)

```bash
cd /p/project1/training2661/$USER          # NOT $HOME (20 GB / 80k file quota)
git clone git@github.com:tum-tt-lab/tt-metal.git
tt-metal/jureca/setup.sh                   # submits build (~10 min) + smoke test
```

Done when `$TT_BASE/logs/tt-train-test-<id>.log` ends with `ALL TT-TRAIN SMOKE TESTS PASSED`.

| File | What |
|---|---|
| `setup.sh` | Login node: init submodules, submit `build.sbatch`, then `test.sbatch` once it succeeds |
| `build.sbatch` | Compute node: hwloc sysroot, venv, `build_metal.sh --build-tt-train`, lib64/ fix, `ttml` .pth files |
| `test.sbatch` | Compute node, 1 chip: import, tiny autograd step, `test_gelu.py` |
| `env.sh` | `source jureca/env.sh` in every job: paths, caches, venv |
| `wandb_sync.sh` | Login node: upload offline wandb runs from `$TT_BASE/runs/*/wandb/` |

**Your own jobs:** copy the `#SBATCH` header of `test.sbatch`, then `source jureca/env.sh`.
`--gres=gpu:N` gives you N chips (Slurm sets `TT_VISIBLE_DEVICES`).

**Rules:** build/run only on compute nodes (Rocky 10; `import ttnn` fails on login nodes).
Use `-c 16` for builds, `-c 4`-`8` for runs. Never reset a chip — ask the maintainers. Wrap binaries
in `timeout`; use `pytest --tb=native`.

**After `git pull` / C++ changes:** `jureca/setup.sh` again (the build is incremental). Python-only
changes need no rebuild.

**Where files go:** pure caches (uv, pip, HuggingFace, torch, kernel cache) in `$TT_CACHE`
(default `/p/scratch/training2661/$USER/.cache`: no backup, 90-day purge, losing it only costs a
re-download). Sysroot, uv Python and CPM sources (build trees use them directly) in `$TT_BASE`
(default `/p/project1/training2661/$USER`). Several clones can share both. PROJECT has a shared
3M-file quota for all of training2661: check it with `jutil project dataquota -p training2661`.

**wandb:** compute nodes have no internet, so jobs log with `WANDB_MODE=offline` into their run
directory (`WANDB_DIR`) and you upload afterwards from a login node: `wandb login` once (key goes to
`~/.netrc`), then `jureca/wandb_sync.sh` after each job. Already-synced runs are skipped.
Runs go to the lab team `tum-tt-lab-runs` (`WANDB_ENTITY` in `env.sh`): ask to be invited there first.
