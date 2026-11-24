# Reference code

Copies of the code that our baseline is compared against, kept here so that every line number
in `../REPORT.md` and in the comments of our code can be looked up. Nothing in this folder is
imported or executed; the test uses the installed timm library (version 1.0.31).

| Folder | What | Source | Fixed version |
|---|---|---|---|
| `deit/` | Training code of the DeiT paper: `main.py`, `engine.py`, `models.py`, `datasets.py`, `requirements.txt` | https://github.com/facebookresearch/deit | commit `7e160fe43f0252d17191b71cbb5826254114ea5b` |
| `vit_jax/` | Model code of the ViT paper: `models_vit.py` | https://github.com/google-research/vision_transformer | commit `64801f1b3b367b3611cc27a3d45cc22870a36fb3` |
| `timm_0.3.3/` | The parts of timm that DeiT calls: scheduler, optimizer factory, model, loss | https://github.com/huggingface/pytorch-image-models | tag `v0.3.3`, commit `1d01c2b68c90619b173d4675d9a283be2bd44a33` |

Downloaded on 24 November 2026. All three projects are published under the Apache License 2.0;
a copy of the licence is in each folder.

Three remarks:

- The Python files have `.txt` added to their names, for example `main.py.txt`. Otherwise this
  repository's formatting hooks would rewrite them and the line numbers would no longer match
  the originals. The hooks still removed spaces at the ends of lines; apart from that the files
  are the originals.
- DeiT's `requirements.txt` asks for timm 0.3.2. That version has no tag in the timm repository,
  so the nearest tagged version, 0.3.3, is stored here.
- The files in `timm_0.3.3/` come from these places in the timm source tree:
  `scheduler/cosine_lr.py`, `scheduler/scheduler.py`, `scheduler/scheduler_factory.py`,
  `optim/optim_factory.py`, `models/vision_transformer.py` and `loss/cross_entropy.py`.

The papers themselves are not stored here: ViT is arXiv:2010.11929, DeiT is arXiv:2012.12877.
