from data_pipeline import (load_hourly_weather, build_subhourly,
                           _idx_to_epoch_sec, _vals_to_epoch_sec)
import config as C
import numpy as np
from datetime import datetime, timezone

def _weather_source(csv_path):
    return [{"path": csv_path,
             "units": {"temp": "F", "pressure": "hPa", "humidity": "%"},
             "columns": {"datetime": ["valid"], "temp": ["tmpf"],
                         "pressure": ["pressure_hpa", "mslp"], "humidity": ["relh"]}}]


def _load_grid(csv_path):
    hourly, raw = load_hourly_weather(_weather_source(csv_path))
    anchor = hourly[C.CHANNELS].to_numpy(dtype=np.float64)
    idx_sec = _idx_to_epoch_sec(hourly.index)
    raw_sec = _vals_to_epoch_sec(raw.index.values)
    raw_vals = [raw[ch].to_numpy(dtype=np.float64) for ch in C.CHANNELS]
    return {"anchor": anchor, "idx_sec": idx_sec, "raw_sec": raw_sec,
            "raw_vals": raw_vals, "n": len(anchor)}


def _slice_window(grid, anchor_pos=None):
    anchor, idx_sec = grid["anchor"], grid["idx_sec"]
    pos = (grid["n"] - 1) if anchor_pos is None else int(anchor_pos)
    if pos + 1 < C.HOURLY_WINDOW_LEN:
        raise RuntimeError(f"only {pos+1} hourly steps; need {C.HOURLY_WINDOW_LEN}")
    win_h = anchor[pos - C.HOURLY_WINDOW_LEN + 1: pos + 1]          
    t_end = int(idx_sec[pos])
    raw_sec = grid["raw_sec"]
    chan_obs = []
    for ci in range(len(C.CHANNELS)):
        v = grid["raw_vals"][ci]
        ok = ~np.isnan(v) & (raw_sec <= t_end)
        chan_obs.append((raw_sec[ok], v[ok]))
    win_s = build_subhourly(chan_obs, t_end)                       

    anchor_dt = datetime.fromtimestamp(t_end, tz=timezone.utc)
    doh = np.zeros(C.DATE_FEAT_DIM, dtype=np.float32)
    doh[anchor_dt.month - 1] = 1.0

    real_h = int(np.isfinite(win_h).all(axis=1).sum())
    quality = {
        "anchor_utc": anchor_dt.isoformat(),
        "hourly_real_slots": real_h,
        "hourly_total_slots": C.HOURLY_WINDOW_LEN,
        "hourly_coverage": round(real_h / C.HOURLY_WINDOW_LEN, 3),
        "subhourly_obs_used": int(sum(len(s) for s, _ in chan_obs)),
        "window_ok": bool(np.isfinite(win_h).all() and np.isfinite(win_s).all()),
    }
    return (win_h.astype(np.float32), win_s.astype(np.float32), doh, quality)


def assemble_windows(csv_path, anchor_pos=None):
    return _slice_window(_load_grid(csv_path), anchor_pos)


def _hourly_positions(csv_path):
    g = _load_grid(csv_path)
    return g["n"], g["idx_sec"]

