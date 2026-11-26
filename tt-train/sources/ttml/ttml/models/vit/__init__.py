# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Vision Transformer (ViT, Dosovitskiy et al. 2020) built from ttml ops.

Design (chosen to keep every tensor tile friendly and avoid ops without autograd):

- Patch embedding = host-side patchify + LinearLayer(C*P*P -> E). This is the same math as the
  stride-P conv in the paper. The input is prepared by :func:`patchify`.
- Pooling (``pool``):
    - ``"cls"``: token row 0 of the input is all zeros, so after the patch Linear it equals
      ``bias + pos[0]``, a free learnable vector, the same as ``cls + pos[0]`` in the paper.
      Head: select token 0 -> LayerNorm.
    - ``"gap"``: no [CLS] token. Head: LayerNorm on every token -> mean over the real tokens
      (big_vision order: the encoder norm runs before pooling).
- Position embedding (``posemb``): ``"learned"`` or fixed ``"sincos2d"`` (big_vision
  ``posemb_sincos_2d``, a buffer, not trained). Padding rows get zeros.
- Sequence padding: the real tokens are padded to a multiple of 32 (tile height). A (1, 1, S, S)
  0/1 key mask in SDPA hides the padded keys, so real tokens never attend to them.
- Encoder blocks: pre-LN (MHA + 4x GELU MLP) with that non-causal mask. ``attention`` picks the
  fused SDPA kernel or the composite (matmul/softmax/matmul) version; ``use_composite_layernorm``
  does the same for LayerNorm. Both exist so optimised and unoptimised variants can be compared.
- Classifier: optional ``head="mlp"`` pre-logits Linear(E -> E) + tanh (big_vision ``rep_size``),
  then a zero-initialised Linear(E -> num_classes).

:meth:`ViTConfig.s16_imagenet` is the "Better plain ViT baselines for ImageNet-1k" (Beyer et al.
2022) model: ViT-S/16, 224px, GAP, sincos2d, MLP head, no dropout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import ttnn
import ttml
from ttml.modules import AbstractModuleBase, Buffer, LinearLayer, ModuleList

from ..nanogpt.gpt_block import LayerNorm
from ..nanogpt.gpt_mlp import GPTMLP
from ..nanogpt.multi_head_attention import MultiHeadAttention
from ..nanogpt.pos_embedding import TrainablePositionalEmbedding
from .autograd_ops import mean_tokens, select_first_token, tanh

TILE = 32

POOLS = ("cls", "gap")
POSEMBS = ("learned", "sincos2d")
HEADS = ("linear", "mlp")
ATTENTIONS = ("fused", "composite")


@dataclass
class ViTConfig:
    """ViT hyperparameters. The defaults are a ViT-Tiny-sized model for 32x32 CIFAR images."""

    image_size: int = 32
    patch_size: int = 4
    in_channels: int = 3
    num_classes: int = 10
    embedding_dim: int = 192
    num_heads: int = 3
    num_blocks: int = 12
    dropout: float = 0.1
    use_composite_layernorm: bool = False
    pool: str = "cls"
    posemb: str = "learned"
    head: str = "linear"
    attention: str = "fused"

    def __post_init__(self) -> None:
        if self.image_size % self.patch_size != 0:
            raise ValueError(f"image_size {self.image_size} not divisible by patch_size {self.patch_size}")
        if self.embedding_dim % self.num_heads != 0:
            raise ValueError(f"embedding_dim {self.embedding_dim} not divisible by num_heads {self.num_heads}")
        for name, allowed in (
            ("pool", POOLS),
            ("posemb", POSEMBS),
            ("head", HEADS),
            ("attention", ATTENTIONS),
        ):
            if getattr(self, name) not in allowed:
                raise ValueError(f"{name}={getattr(self, name)!r}, expected one of {allowed}")
        if self.posemb == "sincos2d" and self.pool == "cls":
            # The [CLS] slot is `bias + pos[0]`; with a fixed pos[0] = 0 it would just be the shared patch bias.
            raise ValueError("posemb='sincos2d' needs pool='gap'")
        if self.posemb == "sincos2d" and self.embedding_dim % 4 != 0:
            raise ValueError(f"sincos2d needs embedding_dim divisible by 4, got {self.embedding_dim}")

    @classmethod
    def s16_imagenet(cls, **overrides) -> "ViTConfig":
        """ViT-S/16 at 224px as in Beyer et al. 2022 (big_vision ``vit_s16_i1k``)."""
        cfg = dict(
            image_size=224,
            patch_size=16,
            num_classes=1000,
            embedding_dim=384,
            num_heads=6,
            num_blocks=12,
            dropout=0.0,
            pool="gap",
            posemb="sincos2d",
            head="mlp",
        )
        cfg.update(overrides)
        return cls(**cfg)

    @property
    def grid_size(self) -> int:
        return self.image_size // self.patch_size

    @property
    def num_patches(self) -> int:
        return self.grid_size**2

    @property
    def patch_dim(self) -> int:
        return self.in_channels * self.patch_size**2

    @property
    def first_patch_row(self) -> int:
        """Row of the first patch token: 1 with a [CLS] slot at row 0, else 0."""
        return 1 if self.pool == "cls" else 0

    @property
    def seq_len(self) -> int:
        """Real tokens: [CLS] (if pool='cls') + patches."""
        return self.first_patch_row + self.num_patches

    @property
    def padded_seq_len(self) -> int:
        return (self.seq_len + TILE - 1) // TILE * TILE


def patchify(images: np.ndarray, config: ViTConfig) -> np.ndarray:
    """(B, C, H, W) float images -> (B, 1, padded_seq_len, patch_dim) model input.

    With pool='cls', row 0 is the zero [CLS] slot. The patch rows hold the flattened patches
    (C, P, P order, like a conv weight) in row-major grid order; the remaining padding rows are zero.
    """
    B, C, H, W = images.shape
    P = config.patch_size
    g = H // P
    patches = images.reshape(B, C, g, P, g, P).transpose(0, 2, 4, 1, 3, 5).reshape(B, g * g, C * P * P)
    out = np.zeros((B, 1, config.padded_seq_len, config.patch_dim), dtype=np.float32)
    r0 = config.first_patch_row
    out[:, 0, r0 : r0 + g * g, :] = patches
    return out


def posemb_sincos_2d(grid: int, dim: int, temperature: float = 10_000.0) -> np.ndarray:
    """(grid*grid, dim) fixed 2D sin-cos position embedding, as big_vision ``posemb_sincos_2d``."""
    y, x = np.meshgrid(np.arange(grid), np.arange(grid), indexing="ij")
    omega = np.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature**omega)
    y = np.einsum("m,d->md", y.flatten(), omega)
    x = np.einsum("m,d->md", x.flatten(), omega)
    return np.concatenate([np.sin(x), np.cos(x), np.sin(y), np.cos(y)], axis=1).astype(np.float32)


class ViTAttention(MultiHeadAttention):
    """NanoGPT MultiHeadAttention with a choice of SDPA implementation (no attention dropout)."""

    def __init__(self, embedding_dim: int, num_heads: int, impl: str = "fused") -> None:
        super().__init__(embedding_dim, num_heads, 0.0)
        self.impl = impl

    def forward(self, x: ttml.autograd.Tensor, mask: Optional[ttml.autograd.Tensor] = None) -> ttml.autograd.Tensor:
        qkv = self.qkv_linear(x)
        query, key, value = ttml.ops.multi_head_utils.heads_creation(qkv, self.num_heads)
        if self.impl == "composite":
            attn = ttml.ops.attention.scaled_dot_product_attention_composite(query, key, value, mask)
        else:
            attn = ttml.ops.attention.scaled_dot_product_attention(query, key, value, mask)
        return self.out_linear(ttml.ops.multi_head_utils.heads_fusion(attn))


class EncoderBlock(AbstractModuleBase):
    """Pre-LN transformer block: x + MHA(LN(x)), then x + MLP(LN(x)). Same params as NanoGPT's GPTBlock."""

    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        E = config.embedding_dim
        self.ln1 = LayerNorm(E, True, config.use_composite_layernorm)
        self.attention = ViTAttention(E, config.num_heads, config.attention)
        self.ln2 = LayerNorm(E, True, config.use_composite_layernorm)
        self.mlp = GPTMLP(E, config.dropout)

    def forward(self, x: ttml.autograd.Tensor, mask: ttml.autograd.Tensor) -> ttml.autograd.Tensor:
        x = ttml.ops.binary.add(self.attention(self.ln1(x), mask), x)
        return ttml.ops.binary.add(self.mlp(self.ln2(x)), x)


class ViT(AbstractModuleBase):
    """Vision Transformer classifier. Input from :func:`patchify`, output logits (B, 1, 1, num_classes)."""

    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.config = config
        E = config.embedding_dim
        S = config.padded_seq_len

        self.patch_embed = LinearLayer(
            config.patch_dim, E, True, weight_init=ttml.init.normal(0.0, 0.02), bias_init=ttml.init.zeros()
        )
        if config.posemb == "learned":
            self.pos_emb = TrainablePositionalEmbedding(S, E, config.dropout)
        else:
            table = np.zeros((1, 1, S, E), dtype=np.float32)
            r0 = config.first_patch_row
            table[0, 0, r0 : r0 + config.num_patches] = posemb_sincos_2d(config.grid_size, E)
            self.posemb_sincos = Buffer(ttml.autograd.Tensor.from_numpy(table, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16))
        self.blocks = ModuleList([EncoderBlock(config) for _ in range(config.num_blocks)])
        self.ln_f = LayerNorm(E, True, config.use_composite_layernorm)
        if config.head == "mlp":
            self.pre_logits = LinearLayer(
                E, E, True, weight_init=ttml.init.normal(0.0, E**-0.5), bias_init=ttml.init.zeros()
            )
        self.head = LinearLayer(E, config.num_classes, True, weight_init=ttml.init.zeros(), bias_init=ttml.init.zeros())

        # Key mask: 1 for real tokens, 0 for padding; the same for every query row. A Buffer, so it is
        # not registered as a parameter (a bare autograd tensor attribute would be).
        mask = np.zeros((1, 1, S, S), dtype=np.float32)
        mask[..., : config.seq_len] = 1.0
        self.attn_mask = Buffer(ttml.autograd.Tensor.from_numpy(mask, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16))
        if config.pool == "gap":
            # Mean over the real patch rows as sum(x * w, dim=2): w = 1/N on real rows, 0 on padding.
            w = np.zeros((1, 1, S, E), dtype=np.float32)
            w[0, 0, : config.num_patches] = 1.0 / config.num_patches
            self.gap_weights = Buffer(ttml.autograd.Tensor.from_numpy(w, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16))

    def forward(self, patches: ttml.autograd.Tensor) -> ttml.autograd.Tensor:
        x = self.patch_embed(patches)  # (B, 1, S, E)
        if self.config.posemb == "learned":
            x = self.pos_emb(x)  # + pos (incl. [CLS] at row 0), dropout
        else:
            x = ttml.ops.binary.add(x, self.posemb_sincos.tensor.get_value())
        mask = self.attn_mask.tensor
        for block in self.blocks:
            x = block(x, mask)
        if self.config.pool == "cls":
            x = self.ln_f(select_first_token(x))  # (B, 1, 1, E)
        else:
            x = mean_tokens(self.ln_f(x), self.gap_weights.tensor.get_value())  # (B, 1, 1, E)
        if self.config.head == "mlp":
            x = tanh(self.pre_logits(x))
        return self.head(x)  # (B, 1, 1, num_classes)


def create_vit(config: ViTConfig) -> ViT:
    return ViT(config)


__all__ = ["ViT", "ViTConfig", "create_vit", "patchify", "posemb_sincos_2d", "select_first_token"]
