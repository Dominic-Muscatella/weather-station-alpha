"""
inference.py
============
Prediction + confidence, exactly as specified:

  1. Monte-Carlo dropout: MC_PASSES (100) forward passes with dropout ON.
     For each pass, per class, threshold the sigmoid at MC_THRESHOLD (0.15) ->
     0/1. Average over passes -> "mc_exceedance" = fraction of passes in which
     the class cleared 15%  (~ "probability the event is >15% likely").
  2. Raw pass: dropout OFF -> plain sigmoid probabilities.
  3. final = (MC_WEIGHT * mc_exceedance + RAW_WEIGHT * raw) / (MC_WEIGHT+RAW_WEIGHT)
     i.e. weighted 2:1, MC : raw.
  4. Confidence interval: the MC_CI percentile band of the MC sigmoid samples.

Everything is per-class over ALL_LABELS (7 events + 'none').
"""
from __future__ import annotations
from typing import Optional
import numpy as np
import torch

import config as C
import units
from engine.model  import DualLegConvNet, enable_mc_dropout


def normalize(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Z-score the 3 channels. X is (..., 3)."""
    return (X - mean) / std


def to_canonical_window(X: np.ndarray, input_units: dict) -> np.ndarray:
    """Convert a (..., 3) array whose last axis is [temp, pressure, humidity]
    from `input_units` (per-channel) into canonical units, in place-safe copy."""
    X = np.asarray(X, dtype=np.float64).copy()
    for j, ch in enumerate(C.CHANNELS):
        u = input_units.get(ch, units.CANONICAL[ch])
        X[..., j] = units.to_canonical(ch, X[..., j], u)
    return X


@torch.no_grad()
def predict(model: DualLegConvNet,
            x_hourly: torch.Tensor, x_sub: torch.Tensor, date_oh: torch.Tensor,
            mc_passes: int = C.MC_PASSES, threshold: float = C.MC_THRESHOLD,
            mc_weight: float = C.MC_WEIGHT, raw_weight: float = C.RAW_WEIGHT,
            ci=C.MC_CI) -> dict:
    """
    Inputs are batched, ALREADY normalized tensors:
        x_hourly (B,168,3), x_sub (B,192,3), date_oh (B,43)
    Returns a dict of (B, 8) numpy arrays:
        final, mc_exceedance, raw, mc_mean, ci_low, ci_high
    """
    device = next(model.parameters()).device
    x_hourly = x_hourly.to(device); x_sub = x_sub.to(device); date_oh = date_oh.to(device)

    # --- Monte-Carlo passes (dropout ON) ---
    enable_mc_dropout(model)
    samples = []
    for _ in range(mc_passes):
        p = torch.sigmoid(model(x_hourly, x_sub, date_oh))
        samples.append(p.unsqueeze(0))
    samples = torch.cat(samples, dim=0)              # (P, B, 8)
    exceed = (samples > threshold).float().mean(dim=0)        # (B, 8)
    mc_mean = samples.mean(dim=0)                             # (B, 8)
    lo = torch.quantile(samples, ci[0] / 100.0, dim=0)
    hi = torch.quantile(samples, ci[1] / 100.0, dim=0)

    # --- Raw pass (dropout OFF) ---
    model.eval()
    raw = torch.sigmoid(model(x_hourly, x_sub, date_oh))      # (B, 8)

    final = (mc_weight * exceed + raw_weight * raw) / (mc_weight + raw_weight)

    return {
        "final": final.cpu().numpy(),
        "mc_exceedance": exceed.cpu().numpy(),
        "raw": raw.cpu().numpy(),
        "mc_mean": mc_mean.cpu().numpy(),
        "ci_low": lo.cpu().numpy(),
        "ci_high": hi.cpu().numpy(),
    }


class WeatherPredictor:
    """Loads a saved bundle and runs both horizons. Used by the API."""
    def __init__(self, package_dir: str, device: str = "cpu"):
        import json, os
        self.device = torch.device(device)
        with open(os.path.join(package_dir, "config.json")) as f:
            self.meta = json.load(f)
        sc = np.load(os.path.join(package_dir, "scaler.npz"))
        self.mean, self.std = sc["mean"], sc["std"]
        self.labels = self.meta["labels"]

        self.model_1h = DualLegConvNet().to(self.device)
        self.model_24h = DualLegConvNet().to(self.device)
        self.model_1h.load_state_dict(torch.load(os.path.join(package_dir, "model_1h.pt"),
                                                 map_location=self.device))
        self.model_24h.load_state_dict(torch.load(os.path.join(package_dir, "model_24h.pt"),
                                                  map_location=self.device))

    def _tensors(self, X, X2, date_oh, input_units=None):
        # Convert live readings to canonical units first (if caller gave units),
        # THEN z-score with the training scaler (which is in canonical units).
        if input_units:
            X = to_canonical_window(X, input_units)
            X2 = to_canonical_window(X2, input_units)
        Xn = normalize(np.asarray(X, np.float32), self.mean, self.std)
        X2n = normalize(np.asarray(X2, np.float32), self.mean, self.std)
        if Xn.ndim == 2:   # single sample -> add batch dim
            Xn, X2n, date_oh = Xn[None], X2n[None], np.asarray(date_oh)[None]
        return (torch.tensor(Xn, dtype=torch.float32),
                torch.tensor(X2n, dtype=torch.float32),
                torch.tensor(np.asarray(date_oh, np.float32)))

    def predict_1h(self, X, X2, date_oh, input_units=None, **kw):
        return self._fmt(predict(self.model_1h,
                                 *self._tensors(X, X2, date_oh, input_units), **kw))

    def predict_24h(self, X, X2, date_oh, input_units=None, **kw):
        return self._fmt(predict(self.model_24h,
                                 *self._tensors(X, X2, date_oh, input_units), **kw))

    def _fmt(self, out: dict):
        """Reshape arrays into a list (per sample) of {label: {...}} dicts."""
        B = out["final"].shape[0]
        results = []
        for b in range(B):
            row = {}
            for j, lab in enumerate(self.labels):
                row[lab] = {
                    "score": float(out["final"][b, j]),
                    "mc_exceedance": float(out["mc_exceedance"][b, j]),
                    "raw_prob": float(out["raw"][b, j]),
                    "ci_low": float(out["ci_low"][b, j]),
                    "ci_high": float(out["ci_high"][b, j]),
                }
            results.append(row)
        return results[0] if B == 1 else results
