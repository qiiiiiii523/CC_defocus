"""NumPy-only diagnostics for soft light/medium/heavy mixture fields."""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

CHANNELS = ("light", "medium", "heavy")
MASK_RULE = {
    "source": "paired_clear_RGB_01; heuristic, not segmentation ground truth",
    "background": "min(R,G,B)>=0.90 and HSV_saturation<=0.10",
    "tissue": "mean(R,G,B)<=0.85 or HSV_saturation>=0.15",
    "uncertain": "all remaining pixels; excluded from tissue/background",
    "grid_weights": "exact block-area fraction, not a hard label per token",
}


def region_masks(clear):
    rgb = np.asarray(clear, dtype=np.float64)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or not np.isfinite(rgb).all():
        raise ValueError("Expected finite HWC RGB")
    if rgb.min() < 0 or rgb.max() > 1:
        raise ValueError("Expected RGB in [0,1]")
    maximum, minimum = rgb.max(-1), rgb.min(-1)
    saturation = (maximum - minimum) / np.maximum(maximum, 1e-12)
    background = (minimum >= .90) & (saturation <= .10)
    tissue = (rgb.mean(-1) <= .85) | (saturation >= .15)
    assert not (background & tissue).any()
    return {"all": np.ones(rgb.shape[:2], dtype=bool), "tissue": tissue,
            "background": background, "uncertain": ~(background | tissue)}


def area_weights(mask, grid):
    """Preserve a region's pixel fraction exactly on integer-sized grids."""
    mask = np.asarray(mask, dtype=np.float64)
    gh, gw = grid
    h, w = mask.shape
    if gh < 1 or gw < 1 or h % gh or w % gw:
        raise ValueError("Region grid must divide the source image exactly")
    return mask.reshape(gh, h // gh, gw, w // gw).mean(axis=(1, 3))


def weighted_channel_stats(pred, truth, weights):
    p, t, w = [np.asarray(x, dtype=np.float64) for x in (pred, truth, weights)]
    if p.shape != t.shape or p.shape != w.shape or p.ndim != 2:
        raise ValueError("Fields and region weights must have the same 2D shape")
    if not all(np.isfinite(x).all() for x in (p, t, w)) or (w < 0).any():
        raise ValueError("Fields and weights must be finite; weights nonnegative")
    total = float(w.sum())
    if total <= 0:
        return None
    def mean(x):
        return float(np.sum(x * w) / total)
    pm, tm = mean(p), mean(t)
    pc, tc, error = p - pm, t - tm, p - t
    ps, ts = math.sqrt(mean(pc ** 2)), math.sqrt(mean(tc ** 2))
    corr = mean(pc * tc) / (ps * ts) if min(ps, ts) > 1e-8 else None
    def tv(x):
        # Only edges with some region support on both ends contribute.
        xw = np.minimum(w[:, 1:], w[:, :-1])
        yw = np.minimum(w[1:, :], w[:-1, :])
        denom = float(xw.sum() + yw.sum())
        return (float((np.abs(np.diff(x, axis=1)) * xw).sum()
                      + (np.abs(np.diff(x, axis=0)) * yw).sum()) / denom
                if denom > 0 else None)
    mae = mean(np.abs(error))
    mean_pred_mae = mean(np.abs(pm - t))
    return {
        "weight_sum": total, "region_fraction": total / w.size,
        "mae": mae, "rmse": math.sqrt(mean(error ** 2)), "bias": mean(error),
        "pred_mean": pm, "truth_mean": tm,
        "pred_spatial_std": ps, "truth_spatial_std": ts,
        "spatial_corr": float(np.clip(corr, -1, 1)) if corr is not None else None,
        "centered_rmse": math.sqrt(mean((pc - tc) ** 2)),
        "mean_pred_mae": mean_pred_mae,
        "spatial_mae_gain": mean_pred_mae - mae,
        "mean_truth_mae": mean(np.abs(tm - t)),
        "pred_tv": tv(p), "truth_tv": tv(t),
    }


def field_rows(pred, truth, masks, *, sample_id, resolution):
    if pred.shape != truth.shape or pred.ndim != 3 or pred.shape[0] != 3:
        raise ValueError("Expected matching [3,H,W] probability fields")
    rows = []
    for region in ("all", "tissue", "background"):
        weights = area_weights(masks[region], pred.shape[-2:])
        for c, channel in enumerate(CHANNELS):
            stats = weighted_channel_stats(pred[c], truth[c], weights)
            if stats is not None:
                rows.append(dict(sample_id=sample_id, resolution=resolution,
                                 region=region, channel=channel, **stats))
    return rows


def aggregate_rows(rows):
    """Report equal-image means and valid counts; undefined correlations stay null."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["resolution"], row["region"], row["channel"])].append(row)
    result = {}
    labels = {"sample_id", "resolution", "region", "channel"}
    for (resolution, region, channel), entries in sorted(grouped.items()):
        stats = {"image_count": len(entries)}
        for key in entries[0].keys() - labels:
            values = [float(r[key]) for r in entries if r[key] is not None]
            stats[key] = {"mean": float(np.mean(values)) if values else None,
                          "valid_images": len(values)}
        result.setdefault(resolution, {}).setdefault(region, {})[channel] = stats
    return result


def compression_rows(pred, truth, pred_grid, truth_grid, pred_rebuilt, truth_rebuilt,
                     *, sample_id):
    rows = []
    for c, channel in enumerate(CHANNELS):
        row = {"sample_id": sample_id, "channel": channel}
        for name, native, grid, rebuilt in (("pred", pred, pred_grid, pred_rebuilt),
                                            ("truth", truth, truth_grid, truth_rebuilt)):
            ns = float(np.std(native[c].astype(np.float64)))
            gs = float(np.std(grid[c].astype(np.float64)))
            row.update({name + "_std_native": ns, name + "_std_token": gs,
                        name + "_std_ratio": gs / ns if ns > 1e-8 else None,
                        name + "_roundtrip_mae": float(np.abs(native[c] - rebuilt[c]).mean())})
        rows.append(row)
    return rows
