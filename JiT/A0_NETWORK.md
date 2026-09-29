# JiT A0 network adaptation

This branch contains the A0 conditional-restoration network plus its dedicated
paired-manifest training and fixed-validation entry point.  It intentionally
does not add a PSF estimator, PSF loss, nucleus losses, or A1-A6 features.

## Network contract

- `clear`: paired clear target used to construct the flow state during training,
  shape `[B, 3, H, W]` in the same normalized range expected by official JiT.
- `blur`: spatially aligned blurred-image condition, same shape and range.
- `degradation`: reserved for A1-A3 and must be `None` in A0.

Training-side call:

```python
loss = denoiser(clear, blur, degradation=None)
```

Inference-side call:

```python
restored = denoiser.generate(blur, degradation=None)
```

## Loading the official checkpoint

The official checkpoint has no blurred-image condition branch.  Load it with
the explicit helper, which validates that only A0 keys are missing and then
copies the pretrained state patch embedding into the new condition encoder:

```python
checkpoint = torch.load(path, map_location="cpu", weights_only=False)
denoiser.load_official_state_dict(checkpoint["model"])
```

Do not replace this with an unchecked `strict=False` load.

## A0 condition path

The original `x_embedder` continues to encode the flow state.  A separate
`BlurConditionEncoder` encodes the aligned blurred image into one token per
state token.  A learnable zero-initialized gate injects these tokens as a
residual, preserving the pretrained JiT behavior at initialization.

The original class embedding is retained for checkpoint compatibility, but A0
always selects JiT's unconditional/null class.  The blurred image is the real
restoration condition and is held fixed throughout ODE sampling.

## A0 task entry point

Use ``main_restoration.py`` rather than the official ImageNet
``main_jit.py``.  The restoration entry point:

- requires exactly 2,000 records from ``debug_train.jsonl``;
- requires exactly 300 records from ``debug_val.jsonl``;
- supports ``--overfit_samples 1..32`` while still validating that the source
  training manifest is the fixed 2,000-pair manifest;
- applies synchronized training crop and horizontal flip;
- optionally converts a configurable fraction of training pairs to
  clear-to-same-clear identity pairs;
- initializes from the official JiT checkpoint or strictly resumes an A0
  checkpoint;
- writes predictions through the shared ``common_io.save_prediction``;
- derives validation noise from ``sample_id`` and ``--eval_seed`` so results
  do not change merely because batch size or distributed world size changes;
- can invoke the shared root ``evaluate.py`` metrics.

No local execution or GPU validation was performed when this integration was
written.  Run the small-pair overfit check on the configured GPU server before
starting the full 2,000-pair experiment.
