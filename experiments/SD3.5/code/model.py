"""Untrained SD3.5 scratch restoration model.

The Transformer is created from the official config and is intentionally NOT
loaded with pretrained weights.  This file has not been GPU-verified locally.
"""
from __future__ import annotations

import torch
from torch import nn


class SD35ScratchRestorer(nn.Module):
    def __init__(self, model_id: str = "stabilityai/stable-diffusion-3.5-medium"):
        super().__init__()
        from diffusers import SD3Transformer2DModel

        cfg = SD3Transformer2DModel.load_config(model_id, subfolder="transformer")
        self.transformer = SD3Transformer2DModel.from_config(cfg)
        in_channels = int(getattr(cfg, "in_channels", 16))
        self.condition_stem = nn.Conv2d(in_channels * 2, in_channels, 3, padding=1)
        self.in_channels = in_channels
        self.joint_attention_dim = int(getattr(cfg, "joint_attention_dim", 4096))
        self.pooled_projection_dim = int(getattr(cfg, "pooled_projection_dim", 2048))

    def forward(self, x_t, z_blur, timestep, encoder_hidden_states=None,
                pooled_projections=None):
        h = self.condition_stem(torch.cat([x_t, z_blur], dim=1))
        b = h.shape[0]
        if encoder_hidden_states is None:
            encoder_hidden_states = h.new_zeros((b, 77, self.joint_attention_dim))
        if pooled_projections is None:
            pooled_projections = h.new_zeros((b, self.pooled_projection_dim))
        out = self.transformer(
            hidden_states=h,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            return_dict=False,
        )
        return out[0] if isinstance(out, tuple) else out.sample
