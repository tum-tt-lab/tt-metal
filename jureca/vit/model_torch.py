"""The ViT in plain PyTorch: the reference model the Tenstorrent version has to match."""

import math

import torch
import torch.nn as nn

from common import NUM_CLASSES, NUM_PATCHES, PATCH_SIZE


class Attention(nn.Module):
    """Multi-head self-attention, no mask: every patch attends to every patch."""

    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)  # q, k and v in one matmul
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).view(B, N, 3, self.heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # each (B, heads, N, head_dim)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)  # (B, heads, N, N)
        out = scores.softmax(dim=-1) @ v  # (B, heads, N, head_dim)
        out = out.transpose(1, 2).reshape(B, N, D)  # concatenate heads
        return self.proj(out)


class Block(nn.Module):
    """Pre-norm transformer encoder block."""

    def __init__(self, dim, heads, mlp_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_dim), nn.GELU(), nn.Linear(mlp_dim, dim))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViT(nn.Module):
    def __init__(self, depth, dim, heads, mlp_dim):
        super().__init__()
        self.patch_embed = nn.Linear(
            3 * PATCH_SIZE * PATCH_SIZE, dim
        )  # 48 -> D. This is the convolution of the 4x4 patches with 3 color depth.
        # 1 to broadcast over batches. For each of the NUM_PATCHES tokens, a pos vector with dim dim is learned.
        self.pos_embed = nn.Parameter(torch.zeros(1, NUM_PATCHES, dim))
        # Create the Transformer Blocks
        self.blocks = nn.Sequential(*[Block(dim, heads, mlp_dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.head = nn.Linear(dim, NUM_CLASSES)

        # BERT initialisation
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    def forward(self, images):  # images: (B, 3, 32, 32)
        x = patchify(images)  # (B, 64, 48)
        x = self.patch_embed(x) + self.pos_embed  # (B, 64, D)
        x = self.blocks(x)
        x = self.norm(x)
        x = x.mean(dim=1)  # global average pooling over the 64 tokens -> (B, D)
        return self.head(x)  # (B, 100)


def patchify(images):
    """(B, 3, 32, 32) -> (B, 64, 48): cut into 4x4 patches, flatten each patch."""
    B, C, H, W = images.shape
    p = PATCH_SIZE
    x = images.view(B, C, H // p, p, W // p, p)  # (B, C, 8, 4, 8, 4)
    x = x.permute(0, 2, 4, 3, 5, 1)  # (B, 8, 8, 4, 4, C): patch row, patch col, pixel row, pixel col, channel
    return x.reshape(B, (H // p) * (W // p), p * p * C)
