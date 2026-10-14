"""Controlled S0/S1/S2 continuation on fixed synthetic known-field pairs."""
from __future__ import annotations
import json
from functools import partial
from pathlib import Path


def make_model(args):
    from denoiser import Denoiser
    model = Denoiser(args)
    model._exp_condition_stats = {}
    def hook(name):
        def capture(module, inputs, output):
            if getattr(model, "_exp_record_condition", False):
                model._exp_condition_stats[name] = float(output.detach().float().square().mean().sqrt().item())
        return capture
    model.net.x_embedder.register_forward_hook(hook("state_token_rms"))
    model.net.blur_condition_encoder.register_forward_hook(hook("rgb_residual_rms"))
    if model.net.local_degradation_condition is not None:
        model.net.local_degradation_condition.register_forward_hook(hook("local_residual_rms"))
    return model


def main():
    import main_restoration as base
    from a1_synthetic_data import SyntheticPairedDataset, sha256_file
    from a1_synthetic_metrics import run_metrics
    parser = base.get_args_parser()
    parser.description = __doc__
    parser.set_defaults(identity_ratio=0, epochs=10, warmup_epochs=2, lr=1e-4,
                        min_lr=1e-6, lr_schedule="cosine", batch_size=8,
                        clip_grad=1, eval_weights="ema2", sampling_method="euler",
                        num_sampling_steps=1, save_last_freq=1, num_workers=4,
                        expected_train_pairs=0, expected_val_pairs=300)
    parser.add_argument("--generation_config", type=Path, required=True)
    parser.add_argument("--data_root", type=Path, default=base.PROJECT_ROOT,
                        help="Explicit root containing the original prepared data and synthetic dataset")
    args = parser.parse_args()
    if args.degradation_mode not in ("none", "oracle_local", "predicted_local"):
        parser.error("Select none=S0, predicted_local=S1, or oracle_local=S2 explicitly")
    if args.identity_ratio != 0 or args.overfit_samples:
        parser.error("Use identity_ratio=0 and the same complete selected dataset for all arms")
    if args.a0_checkpoint is None and args.resume is None:
        parser.error("Use the common original A0 checkpoint, or resume this experiment")
    config = json.loads(args.generation_config.read_text(encoding="utf-8"))
    if config.get("status") != "complete":
        parser.error("Synthetic data generation has not completed")
    for split, path in (("train", args.train_manifest), ("val", args.val_manifest)):
        if sha256_file(path) != config["manifest_sha256"][split]:
            parser.error(f"{split} manifest differs from generation_config")
    if config["image_size"] != args.img_size:
        parser.error("Generated image size differs from model input size")
    args.expected_train_pairs = config["train_count"]
    args.expected_val_pairs = config["val_count"]
    args.synthetic_experiment = True
    args.evaluation_domain = "synthetic_known_gaussian_mixture"
    args.condition_source = {"none": "disabled", "predicted_local": "frozen_v3_prediction",
                             "oracle_local": "actual_generation_field"}[args.degradation_mode]
    args.generation_config_sha256 = sha256_file(args.generation_config)
    args.synthetic_source_group_overlap = config["source_group_overlap"]
    base.main(args, dataset_class=partial(SyntheticPairedDataset, data_root=args.data_root),
              metrics_runner=run_metrics, model_factory=make_model)


if __name__ == "__main__":
    main()
