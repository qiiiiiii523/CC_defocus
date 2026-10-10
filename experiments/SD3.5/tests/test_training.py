"""CPU regressions using a real reduced SD3 Transformer and a deterministic VAE."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))
# Repository common_io is discovered normally after deployment. Staging tests
# pass the original repository explicitly via PYTHONPATH.
from data import PairedManifest
from flow_matching import make_training_state, sample
from infer_scratch import restore_image
from model import SD35ScratchRestorer
from prepare_full_manifests import prepare
from runtime_utils import (capture_rng, check_resume, load_checkpoint, normalize_config,
                           restore_rng, save_checkpoint, seed_all)
from train_scratch import train_epoch, validate
from train_scratch import run_training
from runtime_utils import prune_checkpoints


TINY_CONFIG = {
    "sample_size": 4, "patch_size": 2, "in_channels": 16, "out_channels": 16,
    "num_layers": 2, "attention_head_dim": 8, "num_attention_heads": 2,
    "joint_attention_dim": 32, "caption_projection_dim": 16,
    "pooled_projection_dim": 16, "pos_embed_max_size": 8,
    "dual_attention_layers": [0], "qk_norm": "rms_norm",
}


class TinyVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.config = SimpleNamespace(shift_factor=0.0609, scaling_factor=1.5305)

    def encode(self, x):
        assert x.dtype == torch.float32
        z = torch.nn.functional.avg_pool2d(x, 8).repeat(1, 6, 1, 1)[:, :16]
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: z))

    def decode(self, z):
        assert z.dtype == torch.float32
        x = torch.nn.functional.interpolate(z[:, :3], scale_factor=8, mode="nearest")
        return SimpleNamespace(sample=x)


class Pairs(Dataset):
    def __init__(self, count=5):
        gen = torch.Generator().manual_seed(99)
        self.images = torch.rand(count, 3, 32, 32, generator=gen) * 2 - 1

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return {"blur": self.images[i], "clear": self.images[i] * 0.5, "sample_id": str(i)}


def config(**kwargs):
    return normalize_config(dict({"train_manifest": "train.jsonl", "val_manifest": "val.jsonl",
                                  "resolution": 32, "bf16": False, "max_train_steps": 99,
                                  "gradient_accumulation_steps": 2, "max_grad_norm": 1e6,
                                  "learning_rate": 1e-3}, **kwargs))


def loader(dataset, batch_size=1, seed=11, workers=0):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers,
                      generator=torch.Generator().manual_seed(seed))


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(1)
    # This Windows CPU's oneDNN BF16 backward kernel is unsupported. The native
    # CPU kernels still exercise autocast and autograd without changing the app.
    enabled = torch.backends.mkldnn.enabled
    torch.backends.mkldnn.enabled = False
    try:
        yield
    finally:
        torch.backends.mkldnn.enabled = enabled


def test_actual_transformer_backward_checkpointing_and_bf16():
    seed_all(5)
    model = SD35ScratchRestorer(transformer_config=TINY_CONFIG)
    model.transformer.enable_gradient_checkpointing()
    z = torch.randn(1, 16, 4, 4)
    state, cond, t, target = make_training_state(z, z * 0.5)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        prediction = model(state, cond, t)
    loss = (prediction.float() - target).square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    for module in (model.condition_stem, model.transformer):
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert any(g.abs().sum() > 0 for g in grads)
    model.eval()
    restored = sample(model, z, steps=2, bf16=True)
    assert restored.dtype == torch.float32 and restored.shape == z.shape
    assert torch.isfinite(restored).all()


def test_accumulation_matches_large_batch_including_partial_group():
    seed_all(7)
    initial = SD35ScratchRestorer(transformer_config=TINY_CONFIG)
    outputs = []
    for batch_size, accumulation in ((1, 2), (2, 1)):
        model = copy.deepcopy(initial)
        opt = torch.optim.SGD(model.parameters(), lr=0.001)
        progress = {"step": 0, "epoch": 0, "next_batch": 0}
        cfg = config(batch_size=batch_size, gradient_accumulation_steps=accumulation,
                     flow_condition_noise_alpha=0.0)
        dataset = Pairs()
        # Fix the same time for all examples; noise alpha is zero.
        original = torch.rand
        def fixed_rand(*args, **kwargs):
            if len(args) == 1 and isinstance(args[0], tuple) and "device" in kwargs:
                return torch.full(args[0], 0.5, **kwargs)
            return original(*args, **kwargs)
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(torch, "rand", fixed_rand)
            train_epoch(model, TinyVAE(), opt, loader(dataset, batch_size), torch.device("cpu"),
                        cfg, progress, lambda *a: None)
        assert progress == {"step": 3, "epoch": 1, "next_batch": 0}
        outputs.append(model.state_dict())
    for key in outputs[0]:
        torch.testing.assert_close(outputs[0][key], outputs[1][key], atol=1e-6, rtol=1e-5)


def test_mid_epoch_checkpoint_resume_matches_uninterrupted(tmp_path):
    seed_all(17)
    initial = SD35ScratchRestorer(transformer_config=TINY_CONFIG)
    cfg = config(max_train_steps=3)
    results = []
    for interrupted in (False, True):
        model = copy.deepcopy(initial)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], foreach=False)
        progress = {"step": 0, "epoch": 0, "next_batch": 0}
        seed_all(123)
        partial_cfg = dict(cfg, max_train_steps=1) if interrupted else cfg
        train_epoch(model, TinyVAE(), opt, loader(Pairs()), torch.device("cpu"), partial_cfg,
                    progress, lambda *a: None)
        if interrupted:
            path = tmp_path / "checkpoint-final.pt"
            save_checkpoint(path, model, opt, partial_cfg, "tiny", progress, "digest")
            checkpoint = load_checkpoint(path)
            check_resume(checkpoint, cfg, "digest", "tiny")
            model = SD35ScratchRestorer(transformer_config=checkpoint["transformer_config"])
            model.load_state_dict(checkpoint["model"])
            opt = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], foreach=False)
            opt.load_state_dict(checkpoint["optimizer"])
            progress = dict(checkpoint["progress"])
            train_epoch(model, TinyVAE(), opt, loader(Pairs()), torch.device("cpu"), cfg,
                        progress, lambda *a: None, resume_rng=checkpoint["rng"])
        assert progress == {"step": 3, "epoch": 1, "next_batch": 0}
        results.append(model.state_dict())
    for key in results[0]:
        torch.testing.assert_close(results[0][key], results[1][key], atol=0, rtol=0)


def make_manifest(root):
    rows = []
    for i, split in enumerate(("train", "train", "val", "test")):
        name = f"sample{i}"
        gen = np.random.default_rng(i)
        image = gen.integers(0, 256, size=(48, 48, 3), dtype=np.uint8)
        row = {"sample_id": name, "source_dataset": "RFN", "quality_status": "pass",
               "split": split, "group_id": f"group-{split}"}
        for key, folder in (("blur_path", "3D"), ("clear_path", "3D_label")):
            row[key] = f"prepared/images/3DHistech/{folder}/{name}.png"
            path = root / row[key]
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(image).save(path)
        rows.append(row)
    path = root / "prepared/manifests/rfn.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return rows


def test_full_manifest_filter_and_stateless_paired_augmentation(tmp_path):
    rows = make_manifest(tmp_path)
    counts, outputs = prepare(tmp_path)
    assert counts == {"train": 2, "val": 1, "test": 1}
    assert [json.loads(l)["split"] for l in outputs["train"].read_text().splitlines()] == ["train"] * 2
    ds = PairedManifest(outputs["train"], tmp_path, size=32, max_samples=1, seed=42)
    ds.set_epoch(3)
    a = ds[0]
    seed_all(999)
    b = ds[0]
    torch.testing.assert_close(a["blur"], a["clear"])
    torch.testing.assert_close(a["blur"], b["blur"], atol=0, rtol=0)
    single = next(iter(loader(ds, workers=0)))["blur"]
    worker = next(iter(loader(ds, workers=1)))["blur"]
    torch.testing.assert_close(single, worker, atol=0, rtol=0)


def test_inference_fp32_vae_bf16_model_and_deterministic_output():
    seed_all(5)
    model = SD35ScratchRestorer(transformer_config=TINY_CONFIG).eval()
    blur = np.random.default_rng(7).random((32, 32, 3), dtype=np.float32)
    a = restore_image(model, TinyVAE(), blur, torch.device("cpu"), 32, 3, 0.23, 777, True)
    b = restore_image(model, TinyVAE(), blur, torch.device("cpu"), 32, 3, 0.23, 777, True)
    np.testing.assert_array_equal(a, b)
    assert a.shape == blur.shape and np.isfinite(a).all() and 0 <= a.min() <= a.max() <= 1
    with pytest.raises(ValueError, match="resizing is disabled"):
        restore_image(model, TinyVAE(), blur, torch.device("cpu"), 256, 3, 0.23, 777, True)


@pytest.mark.parametrize("change", [{"resolution": 24}, {"max_train_steps": 0},
                                   {"gradient_accumulation_steps": 0}, {"flow_condition_noise_alpha": float('nan')}])
def test_invalid_config_rejected(change):
    with pytest.raises(ValueError):
        config(**change)


def test_small_overfit_reduces_loss():
    seed_all(29)
    model = SD35ScratchRestorer(transformer_config=TINY_CONFIG)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=False)
    z = torch.randn(2, 16, 4, 4)
    state, cond, t, target = make_training_state(z, z * 0.5, alpha=0.0)
    losses = []
    for _ in range(30):
        opt.zero_grad(set_to_none=True)
        loss = (model(state, cond, t) - target).square().mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.5


def test_resume_rejects_changed_training_contract(tmp_path):
    cfg = config()
    model = SD35ScratchRestorer(transformer_config=TINY_CONFIG)
    opt = torch.optim.AdamW(model.parameters())
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, opt, cfg, "tiny", {"step": 0, "epoch": 0, "next_batch": 0}, "digest")
    ckpt = load_checkpoint(path)
    with pytest.raises(ValueError, match="manifest changed"):
        check_resume(ckpt, cfg, "another", "tiny")
    with pytest.raises(ValueError, match="gradient_accumulation_steps"):
        check_resume(ckpt, dict(cfg, gradient_accumulation_steps=4), "digest", "tiny")


def test_real_local_vae_training_final_save_resume_and_inference(tmp_path, monkeypatch):
    from diffusers import AutoencoderKL
    import infer_scratch
    rows = make_manifest(tmp_path)
    prepare(tmp_path)
    model_dir = tmp_path / "tiny_sd35"
    model = SD35ScratchRestorer(transformer_config=TINY_CONFIG)
    model.transformer.save_config(model_dir / "transformer")
    vae = AutoencoderKL(
        in_channels=3, out_channels=3, latent_channels=16, sample_size=32,
        block_out_channels=(8, 8, 8, 8), layers_per_block=1, norm_num_groups=4,
        down_block_types=("DownEncoderBlock2D",) * 4,
        up_block_types=("UpDecoderBlock2D",) * 4,
        scaling_factor=1.5305, shift_factor=0.0609, force_upcast=True,
    )
    vae.save_pretrained(model_dir / "vae")
    cfg = config(train_manifest="prepared/manifests/full_train.jsonl", max_train_steps=1,
                 num_workers=0, bf16=True, gradient_checkpointing=True,
                 checkpoint_every=0, validation_every=0, overfit_samples=2)
    args = SimpleNamespace(repo_root=tmp_path, output_dir=tmp_path / "runs",
                           resume=None, model_id=str(model_dir))
    run_training(args, cfg, torch.device("cpu"))
    final = args.output_dir / "checkpoint-final.pt"
    assert final.exists()  # A one-update run must save even with periodic saving off.
    checkpoint = load_checkpoint(final)
    assert checkpoint["step"] == 1 and checkpoint["progress"]["epoch"] == 1
    assert checkpoint["optimizer"]["state"]
    args.resume = final
    run_training(args, dict(cfg, max_train_steps=2), torch.device("cpu"))
    resumed = load_checkpoint(final)
    assert resumed["step"] == 2
    continuous_args = SimpleNamespace(repo_root=tmp_path, output_dir=tmp_path / "continuous",
                                      resume=None, model_id=str(model_dir))
    run_training(continuous_args, dict(cfg, max_train_steps=2), torch.device("cpu"))
    continuous = load_checkpoint(continuous_args.output_dir / "checkpoint-final.pt")
    for key in resumed["model"]:
        torch.testing.assert_close(resumed["model"][key], continuous["model"][key], atol=0, rtol=0)
    image_path = tmp_path / "input.png"
    image = np.random.default_rng(7).integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
    Image.fromarray(image).save(image_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(sys, "argv", ["infer_scratch.py", "--checkpoint", str(final),
                                      "--blur", str(image_path), "--out", "single.png"])
    infer_scratch.main()
    assert Image.open(tmp_path / "single.png").size == (32, 32)
    # A corrupt clear reference proves manifest inference never decodes clear pixels.
    val = rows[2]
    Image.fromarray(image).save(tmp_path / val["blur_path"])
    (tmp_path / val["clear_path"]).write_bytes(b"invalid image, must not be decoded")
    monkeypatch.setattr(sys, "argv", ["infer_scratch.py", "--checkpoint", str(final),
                                      "--manifest", "prepared/manifests/full_val.jsonl",
                                      "--repo-root", str(tmp_path), "--model-name", "test"])
    infer_scratch.main()
    saved = tmp_path / "outputs/test" / (val["sample_id"] + ".png")
    assert Image.open(saved).size == (32, 32)


def test_checkpoint_retention_does_not_delete_final_or_other_files(tmp_path):
    for name in ("checkpoint-1.pt", "checkpoint-2.pt", "checkpoint-3.pt",
                 "checkpoint-final.pt", "checkpoint-backup.pt", "other.pt"):
        (tmp_path / name).write_bytes(b"test")
    prune_checkpoints(tmp_path, 2)
    assert not (tmp_path / "checkpoint-1.pt").exists()
    assert {p.name for p in tmp_path.iterdir()} == {
        "checkpoint-2.pt", "checkpoint-3.pt", "checkpoint-final.pt", "checkpoint-backup.pt", "other.pt"}


def test_validation_preserves_rng_training_mode_and_frozen_vae():
    seed_all(1)
    model = SD35ScratchRestorer(transformer_config=TINY_CONFIG).train()
    vae = TinyVAE()
    cfg = config(sampling_steps=2, validation_samples=2)
    rng = capture_rng()
    score = validate(model, vae, Pairs(), torch.device("cpu"), cfg)
    assert np.isfinite(score) and model.training
    torch.testing.assert_close(torch.get_rng_state(), rng["torch"], atol=0, rtol=0)
    assert all(not p.requires_grad and p.grad is None for p in vae.parameters())
