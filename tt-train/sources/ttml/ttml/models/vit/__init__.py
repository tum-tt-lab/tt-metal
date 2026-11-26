# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Vision Transformer (ViT, Dosovitskiy et al. 2020) built from ttml ops.

Design (chosen to keep every tensor tile friendly and avoid ops without autograd):

- Patch embedding = host-side patchify + LinearLayer(C*P*P -> E). This is the same math as the
  stride-P conv in the paper. The input is prepared by :func:`patchify`.
- [CLS] token: token row 0 of the input is all zeros, so after the patch Linear it equals
  ``bias + pos[0]``, which is a free learnable vector, the same as ``cls + pos[0]`` in the paper.
- Sequence padding: 1 + N patches are padded to a multiple of 32 (tile height). A (1, 1, S, S)
  0/1 key mask in SDPA hides the padded keys, so real tokens never attend to them.
- Encoder blocks are NanoGPT's pre-LN GPTBlock (MHA + 4x GELU MLP) with that non-causal mask.
- Head: select token 0 -> LayerNorm -> LinearLayer (zero-initialised, as in the paper).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import ttnn
import ttml
from ttml.modules import AbstractModuleBase, LinearLayer, ModuleList

from ..nanogpt.gpt_block import GPTBlock, LayerNorm
from ..nanogpt.pos_embedding import TrainablePositionalEmbedding
from .autograd_ops import select_first_token

TILE = 32


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

    def __post_init__(self) -> None:
        if self.image_size % self.patch_size != 0:
            raise ValueError(f"image_size {self.image_size} not divisible by patch_size {self.patch_size}")
        if self.embedding_dim % self.num_heads != 0:
            raise ValueError(f"embedding_dim {self.embedding_dim} not divisible by num_heads {self.num_heads}")

    @property
    def num_patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2

    @property
    def patch_dim(self) -> int:
        return self.in_channels * self.patch_size**2

    @property
    def seq_len(self) -> int:
        """Real tokens: [CLS] + patches."""
        return self.num_patches + 1

    @property
    def padded_seq_len(self) -> int:
        return (self.seq_len + TILE - 1) // TILE * TILE


def patchify(images: np.ndarray, config: ViTConfig) -> np.ndarray:
    """(B, C, H, W) float images -> (B, 1, padded_seq_len, patch_dim) model input.

    Row 0 is the zero [CLS] slot, rows 1..num_patches hold the flattened patches (C, P, P order,
    like a conv weight), and the remaining padding rows are zero.
    """
    B, C, H, W = images.shape
    P = config.patch_size
    g = H // P
    patches = images.reshape(B, C, g, P, g, P).transpose(0, 2, 4, 1, 3, 5).reshape(B, g * g, C * P * P)
    out = np.zeros((B, 1, config.padded_seq_len, config.patch_dim), dtype=np.float32)
    out[:, 0, 1 : 1 + g * g, :] = patches
    return out


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
        self.pos_emb = TrainablePositionalEmbedding(S, E, config.dropout)
        self.blocks = ModuleList(
            [
                GPTBlock(E, config.num_heads, config.dropout, True, config.use_composite_layernorm)
                for _ in range(config.num_blocks)
            ]
        )
        self.ln_f = LayerNorm(E, True, config.use_composite_layernorm)
        self.head = LinearLayer(E, config.num_classes, True, weight_init=ttml.init.zeros(), bias_init=ttml.init.zeros())

        # Key mask: 1 for real tokens ([CLS] + patches), 0 for padding; the same for every query row.
        mask = np.zeros((1, 1, S, S), dtype=np.float32)
        mask[..., : config.seq_len] = 1.0
        self.attn_mask = ttml.autograd.Tensor.from_numpy(mask, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16)

    def forward(self, patches: ttml.autograd.Tensor) -> ttml.autograd.Tensor:
        x = self.patch_embed(patches)  # (B, 1, S, E)
        x = self.pos_emb(x)  # + pos (incl. [CLS] at row 0), dropout
        for block in self.blocks:
            x = block(x, self.attn_mask)
        x = select_first_token(x)  # (B, 1, 1, E)
        x = self.ln_f(x)
        return self.head(x)  # (B, 1, 1, num_classes)


def create_vit(config: ViTConfig) -> ViT:
    return ViT(config)


__all__ = ["ViT", "ViTConfig", "create_vit", "patchify", "select_first_token"]
