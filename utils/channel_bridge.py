"""A small gather-and-broadcast channel bridge for the IdealTSF backbone.

This is an independent bottleneck implementation inspired by cross-variable
modeling in TimeBridge, not a copy of its patching or axial encoder.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .attention import scaled_dot_product_attention


class LightChannelBridge(nn.Module):
    """Gather C channels into M representatives, then broadcast to C channels.

    Inputs: queries/keys [B,C,D], values [B,C,L].
    Output: [B,C,L]. No [B,C,C] attention matrix is constructed.
    Existing IdealTSF projections and the forecasting residual are reused.
    """

    def __init__(self, hidden_dim, num_bridges=16):
        super().__init__()
        if hidden_dim < 1 or num_bridges < 1:
            raise ValueError('hidden_dim and num_bridges must be positive.')
        self.bridge_queries = nn.Parameter(torch.empty(num_bridges, hidden_dim))
        nn.init.normal_(self.bridge_queries, std=hidden_dim ** -0.5)

    def forward(self, queries, keys, values):
        # Pooling is independent of channel order. Both steps see history only.
        gather = torch.softmax(
            self.bridge_queries.unsqueeze(0) @ keys.transpose(-2, -1)
            / math.sqrt(keys.shape[-1]), dim=-1,
        )
        bridge_keys = gather @ keys
        bridge_values = gather @ values
        attention = getattr(F, 'scaled_dot_product_attention', scaled_dot_product_attention)
        return attention(queries, bridge_keys, bridge_values, is_causal=False)
