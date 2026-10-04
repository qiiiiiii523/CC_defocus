"""Training and fixed-manifest restoration loops for JiT A0."""

from __future__ import annotations

import hashlib
import math
import sys
from pathlib import Path

import numpy as np
import torch

import util.lr_sched as lr_sched
import util.misc as misc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common_io import save_prediction  # noqa: E402


def train_one_epoch_a0(
    model,
    model_without_ddp,
    data_loader,
    optimizer,
    device,
    epoch,
    *,
    log_writer=None,
    args=None,
):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        "lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    header = f"A0 train epoch: [{epoch}]"
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(metric_logger.log_every(data_loader, 20, header)):
        lr_sched.adjust_learning_rate(
            optimizer, step / len(data_loader) + epoch, args
        )
        clear = batch["clear"].to(device, non_blocking=True)
        blur = batch["blur"].to(device, non_blocking=True)
        identity_fraction = float(batch["is_identity"].float().mean().item())

        amp_enabled = device.type == "cuda" and args.amp_bf16
        with torch.amp.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=amp_enabled,
        ):
            losses = model(
                clear, blur, degradation=None, return_loss_components=True
            )
            loss = losses["loss"]

        loss_values = {name: float(value.detach().item()) for name, value in losses.items()}
        for name, value in loss_values.items():
            if not math.isfinite(value):
                raise RuntimeError(
                    f"Non-finite A0 {name} at epoch={epoch}, step={step}: {value}"
                )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        model_without_ddp.update_ema()

        metric_logger.update(
            **loss_values,
            identity_fraction=identity_fraction,
            lr=optimizer.param_groups[0]["lr"],
        )
        reduced_losses = {
            name: misc.all_reduce_mean(value) for name, value in loss_values.items()
        }
        if log_writer is not None and step % args.log_freq == 0:
            epoch_1000x = int((step / len(data_loader) + epoch) * 1000)
            for name, value in reduced_losses.items():
                log_writer.add_scalar(f"a0/train_{name}", value, epoch_1000x)
            log_writer.add_scalar(
                "a0/identity_fraction", identity_fraction, epoch_1000x
            )
            log_writer.add_scalar(
                "a0/learning_rate", optimizer.param_groups[0]["lr"], epoch_1000x
            )


@torch.no_grad()
def restore_fixed_validation(
    model_without_ddp,
    data_loader,
    device,
    *,
    model_name: str,
    output_root: str | Path,
    use_ema: bool = True,
    expected_total: int = 300,
    amp_bf16: bool = False,
    eval_seed: int = 0,
) -> int:
    """Generate and save this rank's non-overlapping validation shard."""
    original_parameters = None
    if use_ema:
        if model_without_ddp.ema_params1 is None:
            raise RuntimeError("EMA parameters are not initialized")
        original_parameters = [
            parameter.detach().clone()
            for parameter in model_without_ddp.parameters()
        ]
        for parameter, ema_parameter in zip(
            model_without_ddp.parameters(), model_without_ddp.ema_params1
        ):
            parameter.copy_(ema_parameter)

    model_without_ddp.eval()
    saved_count = 0
    try:
        for batch in data_loader:
            blur = batch["blur"].to(device, non_blocking=True)
            sample_ids = batch["sample_id"]
            expected_hw = (int(blur.shape[-2]), int(blur.shape[-1]))
            fixed_noise = _fixed_validation_noise(
                sample_ids,
                channels=int(blur.shape[1]),
                height=expected_hw[0],
                width=expected_hw[1],
                base_seed=eval_seed,
            ).to(device, non_blocking=True)
            amp_enabled = device.type == "cuda" and amp_bf16
            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=amp_enabled,
            ):
                prediction = model_without_ddp.generate(
                    blur,
                    degradation=None,
                    noise=fixed_noise,
                )

            prediction = prediction.add(1.0).div(2.0).clamp_(0.0, 1.0).cpu()
            for index, sample_id in enumerate(sample_ids):
                image = np.asarray(
                    prediction[index].permute(1, 2, 0).contiguous().numpy(),
                    dtype=np.float32,
                )
                save_prediction(
                    image,
                    model_name=model_name,
                    sample_id=sample_id,
                    expected_hw=expected_hw,
                    output_root=output_root,
                )
                saved_count += 1
    finally:
        if original_parameters is not None:
            for parameter, original_parameter in zip(
                model_without_ddp.parameters(), original_parameters
            ):
                parameter.copy_(original_parameter)

    count_tensor = torch.tensor(saved_count, dtype=torch.long, device=device)
    if misc.is_dist_avail_and_initialized():
        torch.distributed.all_reduce(count_tensor, op=torch.distributed.ReduceOp.SUM)
    total_saved = int(count_tensor.item())
    if total_saved != expected_total:
        raise RuntimeError(
            f"Fixed validation restoration must save {expected_total} images, "
            f"but all ranks saved {total_saved}"
        )
    if misc.is_dist_avail_and_initialized():
        torch.distributed.barrier()
    return total_saved


def _fixed_validation_noise(
    sample_ids,
    *,
    channels: int,
    height: int,
    width: int,
    base_seed: int,
) -> torch.Tensor:
    """Create batch/order/world-size independent standard-normal noise."""
    samples = []
    for sample_id in sample_ids:
        digest = hashlib.sha256(
            f"{base_seed}:{sample_id}".encode("utf-8")
        ).digest()
        seed = int.from_bytes(digest[:8], byteorder="little") & ((1 << 63) - 1)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        samples.append(
            torch.randn(
                channels,
                height,
                width,
                generator=generator,
                dtype=torch.float32,
            )
        )
    return torch.stack(samples, dim=0)


__all__ = ["restore_fixed_validation", "train_one_epoch_a0"]
