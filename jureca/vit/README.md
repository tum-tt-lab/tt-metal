# PyTorch reference baseline

As the reference for our Tenstorrent implementation, we train a Vision Transformer from scratch on CIFAR-100 in plain PyTorch. The baseline is meant to be simple and fully specified, not accurate. The architecture follows [ViT](https://arxiv.org/abs/2010.11929); what the ViT paper leaves open (the smaller model sizes, the initialization, which parameters receive weight decay) follows [DeiT](https://arxiv.org/abs/2012.12877) and its implementation in the [timm library](https://github.com/huggingface/pytorch-image-models).

## Model

Each $32\times32$ image is cut into $4\times4$ patches. The resulting 64 tokens are embedded by a linear layer plus a learned position embedding and pass through 12 pre-norm Transformer blocks (ViT, §3.1). Instead of a class token, we average the output tokens after the final layer normalization; ViT reports that both variants perform similarly well (Appendix D.3), and 64 tokens are a multiple of the 32-element tiles that the hardware computes in. A single linear layer produces the class scores. Weights are drawn from a truncated normal distribution with a standard deviation of 0.02, and biases start at zero. The table lists the model sizes; Small is the project's baseline, and Tiny serves for debugging.

| Model | Layers | Width $D$ | Heads | MLP width | Parameters |
|---|---:|---:|---:|---:|---:|
| Tiny | 12 | 192 | 3 | 768 | 5.4 M |
| Small | 12 | 384 | 6 | 1536 | 21.4 M |
| Base | 12 | 768 | 12 | 3072 | 85.2 M |

Model sizes, from DeiT (Tiny, Small) and ViT (Base). Parameter counts are those of our implementation.

## Training

The recipe is that of DeiT (Table 9)[^table9]: AdamW with weight decay 0.05 on the weight matrices, a peak learning rate of $5\cdot10^{-4} \cdot \frac{\text{batchsize}}{512}$ with linear warmup[^warmup] over five epochs and cosine decay towards $10^{-5}$, and label smoothing of 0.1. It uses no dropout or gradient clipping. Adam $\beta_1 = 0.9$ and $\beta_2 = 0.999$ (ViT, under 4.1, Paragraph Training and Fine-tuning). The batch size (default 512) and the number of epochs (default 200) are configurable.

What neither paper mentions is taken from the DeiT code; `test/REPORT.md` lists these details with their sources.

The ViT paper mentions mean-pooling as a viable alternative to having a CLS token, but mentions that the learning rate then is different. This means, the learning rate from DeiT could be wrong. This can be fixed by having a CLS token, which would make it 65 tokens and thus could mess with the 32x32 tiles of tenstorrent.

## Simplifications

We omit DeiT's augmentation and regularisation (RandAugment, Mixup, CutMix, random erasing, stochastic depth) and use no data augmentation at all. The model is therefore expected to overfit. We accept this because what matters here is that two implementations agree.

For comparison, the reference run saves its initial weights and logs the loss at every step, and both implementations inherit the data order, hyperparameters, and learning-rate schedule from a shared module. The reference runs in 32-bit floating-point, whereas the Tenstorrent runs are planned in BF16, so the two loss curves can only agree approximately.

## Implementation

The implementation should match the ViT implementation of DeiT. It is reimplemented in a simpler form to provide an easier overview. It is directly tested against the timm implementation that DeiT uses; see `test/REPORT.md`.

[^table9]: Note that in the table, DeiT-B is mentioned, but on the page before, it says: "Table 9 indicates the hyper-parameters that we use by default at training time for all our experiments, unless stated otherwise."

[^warmup]: In the ViT paper, it was named under 4.1, Paragraph Training and Fine-tuning "We use a linear learning rate warmup". The DeiT paper does not mention it, but the DeiT code warms up linearly.
