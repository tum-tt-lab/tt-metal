# Test report: does our ViT baseline match the reference?

**Yes.** Every check passes. The remaining differences are around one part in a million,
which is ordinary rounding of 32-bit numbers.

Tested on 24 November 2026 on a JURECA login node (CPU, PyTorch 2.11.0, timm 1.0.31).
The raw output of that run is in [output.txt](output.txt).

## What was compared

Our code is a short rewrite, made to be easy to read. The reference is the code everybody uses.

| | Ours | Reference |
|---|---|---|
| Model | `model_torch.py` | The ViT of the timm library |
| Optimizer | `make_optimizer` in `train_torch.py` | timm's optimizer factory, called as in DeiT's `main.py` L352 |
| Learning rate | `lr_at` in `common.py` | timm's cosine scheduler, created as in DeiT's `main.py` L355 |
| Loss | PyTorch cross-entropy with label smoothing | timm's label-smoothing loss, as in DeiT's `main.py` L363 |
| Training loop | Our real script, `train_torch.py` | A loop written like DeiT's `engine.py` L46-68 |

Both sides load the images with our `common.py` and see the batches in the same order.
That part is shared, so it is not checked independently.

## The checks

"Largest difference" is the worst case found. "Allowed" is the limit at which the test fails.

| # | Check | Largest difference | Allowed |
|---|---|---|---|
| 1 | Normalisation constants are the ones DeiT uses | 0 | 0 |
| 2 | Model: same weights and images give the same class scores (Tiny, Small, Base) | 0.0000004 | 0.00001 |
| 3 | Model: the gradients are the same too (Tiny, Small, Base) | 0.000002 | 0.0001 |
| 4 | Learning rate of every epoch, for three different settings | 0 | 0.000000000001 |
| 5 | Optimizer: weight decay, learning rate, betas and epsilon of each of the 151 parameters | 0 parameters differ | 0 |
| 6 | Training: learning rate of every step | 0.00000008 | 0.000001 |
| 7 | Training: loss of every step | 0.0000009 | 0.00001 |
| 8 | Training: all weights at the end | 0.0000006 | 0.00001 |

Checks 2 and 3 use weights with random noise added to every parameter. Freshly initialised
weights have all biases at 0, which would hide a mistake in how biases are handled.

Checks 6 to 8 run our real training script for 16 steps (ViT-Tiny, 1,024 images, batch size 256,
4 epochs of which 2 are warmup), then repeat the same run with the reference, starting from the
weights our script saved. The losses side by side:

| Step | Learning rate | Loss, ours | Loss, reference |
|---|---|---|---|
| 1 | 0.000001 | 4.636306 | 4.636306 |
| 4 | 0.000001 | 4.620094 | 4.620094 |
| 5 | 0.0001255 | 4.635264 | 4.635264 |
| 8 | 0.0001255 | 4.515632 | 4.515632 |
| 9 | 0.00013 | 4.463341 | 4.463341 |
| 12 | 0.00013 | 4.422336 | 4.422335 |
| 13 | 0.00004515 | 4.360052 | 4.360052 |
| 16 | 0.00004515 | 4.406622 | 4.406621 |

The differences are not exactly zero because the two models do the same arithmetic in a
different order (for example a convolution instead of a linear layer for the patches).

## Does the test notice mistakes?

A test that always says "ok" is worthless. So we made 19 copies of our code, put one deliberate
mistake into each, and ran the test on every copy. **It failed on all 19.**

| Mistake put into the copy | Noticed by |
|---|---|
| LayerNorm epsilon 1e-5 instead of 1e-6 | model, training |
| tanh approximation of GELU | model |
| Tokens averaged before the final LayerNorm instead of after | model, training |
| Attention scores divided by sqrt(D) instead of sqrt(head dimension) | model, training |
| Pixels inside a patch flattened in a different order | model, training |
| No bias on q, k, v | model (the weights no longer fit into the reference) |
| Weight decay also on biases, LayerNorm and position embedding | optimizer, training |
| Weight decay also on the position embedding | optimizer |
| Weight decay 0.04 instead of 0.05 | optimizer |
| Adam epsilon 1e-6 instead of 1e-8 | optimizer, training |
| Adam beta2 0.99 instead of 0.999 | optimizer, training |
| No label smoothing | training |
| Label smoothing as worded in the DeiT paper (0.9 and 0.1/99) | training |
| Learning rate not scaled with the batch size | training |
| Learning rate one epoch late | training |
| Warmup starts at 0 instead of 1e-6 | schedule, training |
| Cosine decays to 0 instead of 1e-5 | schedule, training |
| Cosine counted from the end of warmup instead of from epoch 0 | schedule, training |
| CIFAR mean instead of the ImageNet mean | normalisation |

Two of these (weight decay on the position embedding, and weight decay 0.04) are too small to
show up in 16 training steps. Only check 5, which looks at the optimizer settings directly,
notices them. `broken_copies.py` repeats this whole experiment.

## What the test does not show

- **Initialisation.** The reference always starts from our saved weights, so how we draw the
  initial weights is not compared.
- **Long runs.** Training is compared for 16 steps of ViT-Tiny, not for 200 epochs. The model
  check covers all three sizes.
- **Data loading and batch order.** Both sides use ours.
- **The exact library version of the paper.** The test runs timm 1.0.31. DeiT was written for
  timm 0.3.2. The parts DeiT calls (schedule, weight-decay rule, loss) were compared by reading;
  they are the same. The old files are in `reference/timm_0.3.3/`.
- **DeiT's own script.** It is not executed. Its settings were typed into `deit_args` in the
  test by hand; compare them with `reference/deit/main.py.txt` L56-92.
- **Other hardware or number formats.** Everything ran on a CPU in 32-bit floating point.
- **Whether the training is any good.** The test only shows that two implementations agree.

## Where we deliberately differ from DeiT

1. **Patch size 4 on 32x32 images**, instead of 16 on 224x224.
2. **Mean pooling instead of a class token.** The ViT paper (Appendix D.3, Figure 9) says that
   mean pooling works as well as a class token, but needs a different learning rate. We keep
   DeiT's learning rate, which was chosen for a class token, so it may not be the best one for
   our model. The reference model in the test is set up with mean pooling too, in the way the
   ViT authors' code does it (`reference/vit_jax/models_vit.py.txt` L206 and L289-290).
3. **No augmentation, stochastic depth, weight averaging or mixed precision.**
4. **Batch size 512 and 200 epochs by default.** Neither paper trains from scratch on 32x32
   images, so these defaults are our own choice. Both can be set with `--batch-size` and
   `--epochs`; the learning rate follows the batch size.
5. **The schedule is not one epoch late.** DeiT's `main.py` L432 updates the learning rate
   *after* each epoch with the value for that same epoch, so every epoch trains with the
   previous epoch's rate and the first two epochs both run at 1e-6. The paper says five warmup
   epochs, so we use each epoch's own rate. The reference in the test does the same.

## Where every setting comes from

"Not in the papers" means that neither paper states it, so the DeiT code decides.
Paths in the last column are inside `reference/`.

| Setting | Value | Our code | Paper | Reference code |
|---|---|---|---|---|
| Model sizes | 12 layers; width 192 / 384 / 768; 3 / 6 / 12 heads; MLP 4x width | `common.py` L14-16 | DeiT Table 1, §3; ViT Table 1 | `deit/models.py.txt` L65, L80, L95 |
| Patch embedding | Linear layer on flattened patches | `model_torch.py` L50, L76-82 | ViT §3.1, Eq. 1 | |
| Position embedding | Learned | `model_torch.py` L54 | ViT §3.1 | |
| Block | LayerNorm before attention and before MLP | `model_torch.py` L42-43 | ViT Eq. 2-3 | |
| Attention scale | 1 / sqrt(head dimension) | `model_torch.py` L25 | ViT Appendix A, Eq. 6 | `timm_0.3.3/vision_transformer.py.txt` L104 |
| Bias on q, k, v | Yes | `model_torch.py` L18 | Not in the papers | `deit/models.py.txt` L65 |
| Activation | GELU, exact form | `model_torch.py` L39 | ViT §3.1; DeiT §3 | `timm_0.3.3/vision_transformer.py.txt` L80 |
| LayerNorm epsilon | 1e-6 | `model_torch.py` L36, L38, L57 | Not in the papers | `deit/models.py.txt` L66 |
| Pooling | Mean over tokens, after the final LayerNorm | `model_torch.py` L71-72 | ViT Appendix D.3 | `vit_jax/models_vit.py.txt` L206, L289-290 |
| Classifier | One linear layer | `model_torch.py` L58 | DeiT §5.1 | |
| Optimizer | AdamW | `train_torch.py` L41 | DeiT Table 9 | `deit/main.py.txt` L56 |
| Adam betas | 0.9 and 0.999 | `train_torch.py` L44 | ViT §4.1 | `deit/main.py.txt` L60 |
| Adam epsilon | 1e-8 | `train_torch.py` L45 | Not in the papers | `deit/main.py.txt` L58 |
| Weight decay | 0.05 | `common.py` L55 | DeiT Table 9 | `deit/main.py.txt` L66 |
| No weight decay on | Biases, LayerNorm, position embedding | `train_torch.py` L39-40 | Not in the papers | `timm_0.3.3/optim_factory.py.txt` L31; `timm_0.3.3/vision_transformer.py.txt` L255-256 |
| Peak learning rate | 0.0005 x batch size / 512 | `common.py` L54; `train_torch.py` L67 | DeiT Table 9, §6 | `deit/main.py.txt` L71, L350 |
| Warmup | Linear, 5 epochs | `common.py` L56, L70-71 | DeiT Table 9; ViT §4.1 | `deit/main.py.txt` L86; `timm_0.3.3/cosine_lr.py.txt` L63, L70 |
| Warmup start | 1e-6 | `common.py` L26 | Not in the papers | `deit/main.py.txt` L79 |
| Decay | Cosine, towards 1e-5 | `common.py` L27, L72 | DeiT Table 9 (cosine only) | `deit/main.py.txt` L81; `timm_0.3.3/cosine_lr.py.txt` L90 |
| Learning rate changes | Once per epoch | `train_torch.py` L75 | Not in the papers | `timm_0.3.3/scheduler_factory.py.txt` L35 |
| Label smoothing | 0.1, spread over all classes | `train_torch.py` L83 | DeiT Table 9, §4 | `deit/main.py.txt` L101, L363; `timm_0.3.3/cross_entropy.py.txt` L18-25 |
| Normalisation | ImageNet mean and standard deviation | `common.py` L23-24 | Not in the papers | `deit/datasets.py.txt` L108 |
| Batch order | New random order each epoch, last incomplete batch dropped | `common.py` L42-43 | Not in the papers | `deit/main.py.txt` L233, L241 |
| Dropout, gradient clipping | None | | DeiT Table 9 | `deit/main.py.txt` L44, L62 |
| Batch size, epochs | 512 and 200 by default, configurable | `common.py` L52-53 | Own choice | |

One wording difference: the DeiT paper (§4) describes label smoothing as 0.9 for the true class
and the rest shared among the *other* classes. Its code, and PyTorch, share the rest among *all*
classes (0.901 and 0.001). We follow the code.

## Run it again

From the repository root, after `source jureca/env.sh`:

```
python jureca/vit/test/test_against_reference.py     # the test, about 3 minutes on 4 CPU cores
python jureca/vit/test/broken_copies.py              # the 19 mistakes, about 10 minutes on 20 CPU cores
```
