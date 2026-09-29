# JiT A0 network adaptation

This branch contains only the network-side adaptation for A0 conditional
restoration.  It intentionally does not add the paired dataset, training
entry point, evaluation loop, PSF estimator, PSF loss, or nucleus losses.

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
