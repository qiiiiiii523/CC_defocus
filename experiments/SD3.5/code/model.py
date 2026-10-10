"""SD3.5 restoration with a random Transformer and a spatial blur condition."""
import torch
from torch import nn

DEFAULT_MODEL_ID = "stabilityai/stable-diffusion-3.5-medium"


class SD35ScratchRestorer(nn.Module):
    def __init__(self, model_id=DEFAULT_MODEL_ID, transformer_config=None):
        super().__init__()
        from diffusers import SD3Transformer2DModel

        cfg = transformer_config
        if cfg is None:
            cfg = SD3Transformer2DModel.load_config(model_id, subfolder="transformer")
        # Config only: never load pretrained Transformer weights.
        self.transformer = SD3Transformer2DModel.from_config(cfg)
        cfg = self.transformer.config
        self.in_channels = int(cfg.in_channels)
        self.joint_attention_dim = int(cfg.joint_attention_dim)
        self.pooled_projection_dim = int(cfg.pooled_projection_dim)
        self.condition_stem = nn.Conv2d(self.in_channels * 2, self.in_channels, 3, padding=1)

    def forward(self, x_t, z_blur, timestep, encoder_hidden_states=None,
                pooled_projections=None):
        if x_t.ndim != 4 or x_t.shape != z_blur.shape:
            raise ValueError("State and blur condition must have identical BCHW shapes")
        if x_t.shape[1] != self.in_channels:
            raise ValueError("Latent channel count differs from Transformer config")
        h = self.condition_stem(torch.cat([x_t, z_blur], dim=1))
        b = h.shape[0]
        if encoder_hidden_states is None:
            encoder_hidden_states = h.new_zeros((b, 77, self.joint_attention_dim))
        if pooled_projections is None:
            pooled_projections = h.new_zeros((b, self.pooled_projection_dim))
        # SD3 handles patch embedding and unpatchifying internally: BCHW in/out.
        out = self.transformer(
            hidden_states=h, timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections, return_dict=False,
        )
        y = out[0] if isinstance(out, tuple) else out.sample
        if y.shape != x_t.shape:
            raise ValueError("Transformer output {} differs from state {}".format(y.shape, x_t.shape))
        return y
