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
