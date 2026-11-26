# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Autograd ops the ViT needs that ttml does not provide as autograd ops."""

from __future__ import annotations

import ttnn
import ttml


class SelectFirstToken(ttml.autograd.Function):
    """x[:, :, 0:1, :] with autograd: (B, H, S, E) -> (B, H, 1, E).

    Forward: ttnn.slice. Backward: zero-pad the gradient back to S rows.
    """

    @staticmethod
    def forward(ctx, x):
        value = x.get_value()
        B, H, S, E = list(value.shape)
        ctx.seq_len = S
        return ttnn.slice(value, [0, 0, 0, 0], [B, H, 1, E])

    @staticmethod
    def backward(ctx, grad_output):
        return ttnn.pad(grad_output, [(0, 0), (0, 0), (0, ctx.seq_len - 1), (0, 0)], 0.0)


def select_first_token(x: ttml.autograd.Tensor) -> ttml.autograd.Tensor:
    """Return token 0 (the [CLS] token) of a (B, H, S, E) tensor, keeping the S dim."""
    return SelectFirstToken.apply(x)


class MeanTokens(ttml.autograd.Function):
    """Weighted sum over the token dim: (B, H, S, E), weights (1, 1, S, E) -> (B, H, 1, E).

    With weights 1/N on the N real rows and 0 on padding rows this is the mean over the real tokens.
    Forward: sum(x * w, dim=2). Backward: dL/dx = w * dL/dy repeated over S.
    """

    @staticmethod
    def forward(ctx, x, weights):
        value = x.get_value()
        ctx.weights = weights
        ctx.seq_len = list(value.shape)[2]
        return ttnn.sum(ttnn.multiply(value, weights), dim=2, keepdim=True)

    @staticmethod
    def backward(ctx, grad_output):
        return ttnn.multiply(ttnn.repeat(grad_output, [1, 1, ctx.seq_len, 1]), ctx.weights)


def mean_tokens(x: ttml.autograd.Tensor, weights: ttnn.Tensor) -> ttml.autograd.Tensor:
    """Global average pooling over the tokens selected by `weights` (see :class:`MeanTokens`)."""
    return MeanTokens.apply(x, weights)


class Tanh(ttml.autograd.Function):
    """tanh with autograd. Backward: dL/dx = dL/dy * (1 - y^2)."""

    @staticmethod
    def forward(ctx, x):
        y = ttnn.tanh(x.get_value())
        ctx.y = y
        return y

    @staticmethod
    def backward(ctx, grad_output):
        one_minus_y2 = ttnn.add(ttnn.neg(ttnn.multiply(ctx.y, ctx.y)), 1.0)
        return ttnn.multiply(grad_output, one_minus_y2)


def tanh(x: ttml.autograd.Tensor) -> ttml.autograd.Tensor:
    return Tanh.apply(x)
