import torch
import torch.nn as nn
from model_jit import JiT_models
from restoration_losses import charbonnier_loss
from degradation.model import BlurEstimator
from degradation.inference import load_estimator, predict_probabilities


class Denoiser(nn.Module):
    def __init__(
        self,
        args
    ):
        super().__init__()
        self.degradation_mode = getattr(args, "degradation_mode", None) or "none"
        self.net = JiT_models[args.model](
            input_size=args.img_size,
            in_channels=3,
            num_classes=args.class_num,
            attn_drop=args.attn_dropout,
            proj_drop=args.proj_dropout,
            degradation_conditioning=self.degradation_mode != "none",
        )
        self.degradation_estimator = (
            BlurEstimator().eval().requires_grad_(False)
            if self.degradation_mode == "predicted" else None
        )
        self.img_size = args.img_size
        self.num_classes = args.class_num

        self.label_drop_prob = args.label_drop_prob
        self.P_mean = args.P_mean
        self.P_std = args.P_std
        self.t_eps = args.t_eps
        self.noise_scale = args.noise_scale
        # Legacy callers without these options keep the original flow loss.
        self.lambda_pix = getattr(args, "lambda_pix", 0.0)
        self.charbonnier_eps = getattr(args, "charbonnier_eps", 1e-3)

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

    def train(self, mode=True):
        super().train(mode)
        # requires_grad=False alone does not freeze BatchNorm running statistics.
        if self.degradation_estimator is not None:
            self.degradation_estimator.eval()
        return self

    def load_degradation_estimator(self, path):
        if self.degradation_estimator is None:
            raise ValueError("Estimator weights are only used in predicted mode")
        estimator = load_estimator(path)
        self.degradation_estimator.load_state_dict(estimator.state_dict(), strict=True)
        self.degradation_estimator.eval().requires_grad_(False)

    def load_a0_state_dict(self, state_dict):
        """Load all A0 parameters strictly; only the new A1 modules may be absent."""
        if any(name.startswith(("net.degradation_condition.", "degradation_estimator."))
               for name in state_dict):
            raise ValueError("--a0_checkpoint must be an A0 checkpoint, not A1")
        incompatible = self.load_state_dict(state_dict, strict=False)
        invalid = [name for name in incompatible.missing_keys if not name.startswith(
            ("net.degradation_condition.", "degradation_estimator."))]
        if invalid or incompatible.unexpected_keys:
            raise RuntimeError(f"Incompatible A0 checkpoint: {incompatible}")
        return incompatible

    def _degradation_probabilities(self, blur, degradation=None):
        if self.degradation_mode == "none":
            if degradation is not None:
                raise ValueError("Explicit degradation is not allowed in none mode")
            return None
        if degradation is not None:
            raise ValueError("Conditions are produced from the current blur input, not external labels")
        if self.degradation_mode == "fixed":
            probabilities = blur.new_zeros((blur.shape[0], 3))
            probabilities[:, 0] = 1
            return probabilities
        # Use the actual synchronized/identity-replaced RGB condition. FP32
        # keeps this frozen estimator independent of the restoration AMP mode.
        self.degradation_estimator.eval()
        with torch.no_grad(), torch.amp.autocast(device_type=blur.device.type, enabled=False):
            return predict_probabilities(self.degradation_estimator, (blur.float() + 1) * 0.5)

    def load_official_state_dict(self, state_dict):
        """Load an official JiT Denoiser state dict into the A0 network.

        Only the newly introduced blurred-image condition branch may be
        absent.  Any other missing or unexpected key is treated as an error so
        checkpoint incompatibilities cannot be silently ignored.
        """
        incompatible = self.load_state_dict(state_dict, strict=False)
        allowed_missing_prefixes = (
            "net.blur_condition_encoder.", "net.degradation_condition.", "degradation_estimator."
        )
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

    def forward(self, clear, blur, degradation=None, return_loss_components=False):
        """Compute flow + weighted pixel loss on the same sampled endpoint.

        Default return remains a scalar for existing callers. The training
        engine requests components for separate logging.
        """
        if clear.shape != blur.shape:
            raise ValueError(
                f"clear and blur must have identical shapes; got "
                f"clear={tuple(clear.shape)} and blur={tuple(blur.shape)}"
            )
        degradation = self._degradation_probabilities(blur, degradation)

        labels = self._null_labels(clear.size(0), clear.device)
        t = self.sample_t(clear.size(0), device=clear.device).view(
            -1, *([1] * (clear.ndim - 1))
        )
        e = torch.randn_like(clear) * self.noise_scale

        z = t * clear + (1 - t) * e
        v = (clear - z) / (1 - t).clamp_min(self.t_eps)

        x_pred = self.net(z, t.flatten(), labels, blur=blur, degradation=degradation)
        v_pred = (x_pred - z) / (1 - t).clamp_min(self.t_eps)

        # Preserve the official flow objective and add unweighted-in-time
        # RGB pixel supervision of x_pred (not the noisy state z).
        loss_flow = ((v - v_pred) ** 2).mean(dim=(1, 2, 3)).mean()
        loss_charb = charbonnier_loss(x_pred, clear, eps=self.charbonnier_eps)
        loss_pix_weighted = self.lambda_pix * loss_charb
        loss = loss_flow + loss_pix_weighted
        if return_loss_components:
            return {
                "loss": loss,
                "loss_flow": loss_flow,
                "loss_charb": loss_charb,
                "loss_pix_weighted": loss_pix_weighted,
            }
        return loss

    @torch.no_grad()
    def generate(self, blur, degradation=None, noise=None):
        """Restore images from noise while keeping the blur condition fixed."""
        degradation = self._degradation_probabilities(blur, degradation)
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
            z = stepper(z, t, t_next, blur, degradation)
        # last step euler
        z = self._euler_step(z, timesteps[-2], timesteps[-1], blur, degradation)
        return z

    @torch.no_grad()
    def _forward_sample(self, z, t, blur, degradation=None):
        labels = self._null_labels(z.size(0), z.device)
        x_pred = self.net(z, t.flatten(), labels, blur=blur, degradation=degradation)
        return (x_pred - z) / (1.0 - t).clamp_min(self.t_eps)

    @torch.no_grad()
    def _euler_step(self, z, t, t_next, blur, degradation=None):
        v_pred = self._forward_sample(z, t, blur, degradation)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def _heun_step(self, z, t, t_next, blur, degradation=None):
        v_pred_t = self._forward_sample(z, t, blur, degradation)

        z_next_euler = z + (t_next - t) * v_pred_t
        v_pred_t_next = self._forward_sample(z_next_euler, t_next, blur, degradation)

        v_pred = 0.5 * (v_pred_t + v_pred_t_next)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def update_ema(self):
        source_params = list(self.parameters())
        for targ, src in zip(self.ema_params1, source_params):
            if src.requires_grad:
                targ.detach().mul_(self.ema_decay1).add_(src, alpha=1 - self.ema_decay1)
            else:
                targ.copy_(src)
        for targ, src in zip(self.ema_params2, source_params):
            if src.requires_grad:
                targ.detach().mul_(self.ema_decay2).add_(src, alpha=1 - self.ema_decay2)
            else:
                targ.copy_(src)
