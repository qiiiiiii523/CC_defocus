import torch
import torch.nn as nn
from model_jit import JiT_models


class Denoiser(nn.Module):
    def __init__(
        self,
        args
    ):
        super().__init__()
        self.net = JiT_models[args.model](
            input_size=args.img_size,
            in_channels=3,
            num_classes=args.class_num,
            attn_drop=args.attn_dropout,
            proj_drop=args.proj_dropout,
        )
        self.img_size = args.img_size
        self.num_classes = args.class_num

        self.label_drop_prob = args.label_drop_prob
        self.P_mean = args.P_mean
        self.P_std = args.P_std
        self.t_eps = args.t_eps
        self.noise_scale = args.noise_scale

        # ema
        self.ema_decay1 = args.ema_decay1
        self.ema_decay2 = args.ema_decay2
        self.ema_params1 = None
        self.ema_params2 = None

        # generation hyper params
        self.method = args.sampling_method
        self.steps = args.num_sampling_steps
        self.cfg_scale = args.cfg
        self.cfg_interval = (args.interval_min, args.interval_max)

    def _null_labels(self, batch_size, device):
        """Use JiT's pretrained unconditional class embedding for restoration."""
        return torch.full(
            (batch_size,), self.num_classes, dtype=torch.long, device=device
        )

    def load_official_state_dict(self, state_dict):
        """Load an official JiT Denoiser state dict into the A0 network.

        Only the newly introduced blurred-image condition branch may be
        absent.  Any other missing or unexpected key is treated as an error so
        checkpoint incompatibilities cannot be silently ignored.
        """
        incompatible = self.load_state_dict(state_dict, strict=False)
        allowed_missing_prefixes = ("net.blur_condition_encoder.",)
        invalid_missing = [
            key for key in incompatible.missing_keys
            if not key.startswith(allowed_missing_prefixes)
        ]
        if invalid_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                "Official JiT checkpoint is incompatible with the A0 network. "
                f"Unexpected keys: {incompatible.unexpected_keys}; "
                f"invalid missing keys: {invalid_missing}"
            )

        self.net.initialize_blur_condition_from_state_embedder()
        return incompatible

    def sample_t(self, n: int, device=None):
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    def forward(self, clear, blur, degradation=None):
        """Compute the A0 flow-matching loss for a paired clear/blur batch."""
        if clear.shape != blur.shape:
            raise ValueError(
                f"clear and blur must have identical shapes; got "
                f"clear={tuple(clear.shape)} and blur={tuple(blur.shape)}"
            )
        if degradation is not None:
            raise NotImplementedError(
                "degradation conditioning belongs to A1-A3 and is not enabled in A0"
            )

        labels = self._null_labels(clear.size(0), clear.device)
        t = self.sample_t(clear.size(0), device=clear.device).view(
            -1, *([1] * (clear.ndim - 1))
        )
        e = torch.randn_like(clear) * self.noise_scale

        z = t * clear + (1 - t) * e
        v = (clear - z) / (1 - t).clamp_min(self.t_eps)

        x_pred = self.net(z, t.flatten(), labels, blur=blur, degradation=None)
        v_pred = (x_pred - z) / (1 - t).clamp_min(self.t_eps)

        # l2 loss
        loss = (v - v_pred) ** 2
        loss = loss.mean(dim=(1, 2, 3)).mean()

        return loss

    @torch.no_grad()
    def generate(self, blur, degradation=None, noise=None):
        """Restore images from noise while keeping the blur condition fixed."""
        if degradation is not None:
            raise NotImplementedError(
                "degradation conditioning belongs to A1-A3 and is not enabled in A0"
            )
        device = blur.device
        bsz = blur.size(0)
        if noise is None:
            noise = torch.randn(
                bsz, 3, self.img_size, self.img_size, device=device
            )
        else:
            expected_shape = (bsz, 3, self.img_size, self.img_size)
            if tuple(noise.shape) != expected_shape:
                raise ValueError(
                    f"noise must have shape {expected_shape}, got {tuple(noise.shape)}"
                )
            noise = noise.to(device=device, dtype=blur.dtype)
        z = self.noise_scale * noise
        timesteps = torch.linspace(0.0, 1.0, self.steps+1, device=device).view(-1, *([1] * z.ndim)).expand(-1, bsz, -1, -1, -1)

        if self.method == "euler":
            stepper = self._euler_step
        elif self.method == "heun":
            stepper = self._heun_step
        else:
            raise NotImplementedError

        # ode
        for i in range(self.steps - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            z = stepper(z, t, t_next, blur)
        # last step euler
        z = self._euler_step(z, timesteps[-2], timesteps[-1], blur)
        return z

    @torch.no_grad()
    def _forward_sample(self, z, t, blur):
        labels = self._null_labels(z.size(0), z.device)
        x_pred = self.net(z, t.flatten(), labels, blur=blur, degradation=None)
        return (x_pred - z) / (1.0 - t).clamp_min(self.t_eps)

    @torch.no_grad()
    def _euler_step(self, z, t, t_next, blur):
        v_pred = self._forward_sample(z, t, blur)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def _heun_step(self, z, t, t_next, blur):
        v_pred_t = self._forward_sample(z, t, blur)

        z_next_euler = z + (t_next - t) * v_pred_t
        v_pred_t_next = self._forward_sample(z_next_euler, t_next, blur)

        v_pred = 0.5 * (v_pred_t + v_pred_t_next)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def update_ema(self):
        source_params = list(self.parameters())
        for targ, src in zip(self.ema_params1, source_params):
            targ.detach().mul_(self.ema_decay1).add_(src, alpha=1 - self.ema_decay1)
        for targ, src in zip(self.ema_params2, source_params):
            targ.detach().mul_(self.ema_decay2).add_(src, alpha=1 - self.ema_decay2)
