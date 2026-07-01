"""data_pipeline.py — turns raw files into modeling arrays (synced to user copy)."""
from __future__ import annotations
import json
import os
from typing import Optional

import numpy as np
import pandas as pd
import math
import config as C
import units
from event_mapping import map_events_to_classes

TRAIN_FRAC = C.TRAIN_SPLIT
VAL_FRAC = C.VAL_SPLIT

_HEAT_THR_C = (units.to_canonical("temp", C.HEAT_THRESHOLD_F, "F")
               if C.HEAT_THRESHOLD_F is not None else None)
_COLD_THR_C = (units.to_canonical("temp", C.COLD_THRESHOLD_F, "F")
               if C.COLD_THRESHOLD_F is not None else None)
_HEAT_IDX = C.ALL_LABELS.index("severe heat")
_COLD_IDX = C.ALL_LABELS.index("severe cold")


def _resolve_col(df, candidates):
    lut = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        if cand.lower().strip() in lut:
            return lut[cand.lower().strip()]
    return None


def parse_datetime_series(s):
    if pd.api.types.is_datetime64_any_dtype(s):
        return pd.to_datetime(s, utc=True, errors="coerce")
    nonnull = s.dropna()
    if len(nonnull):
        as_str = nonnull.astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
        if as_str.str.fullmatch(r"\d+").all():
            L = as_str.str.len().mode().iloc[0]
            fmt = {12: "%Y%m%d%H%M", 14: "%Y%m%d%H%M%S",
                   10: "%Y%m%d%H", 8: "%Y%m%d"}.get(int(L))
            if fmt:
                full = s.astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
                return pd.to_datetime(full, format=fmt, utc=True, errors="coerce")
    out = pd.to_datetime(s, format="%d-%b-%y %H:%M:%S", utc=True, errors="coerce")
    if out.notna().mean() > 0.5:
        return out
    return pd.to_datetime(s, utc=True, errors="coerce")


def _assemble_datetime(df):
    parts = {}
    for key, cands in C.WEATHER_DATETIME_PARTS.items():
        col = _resolve_col(df, cands)
        if col is not None:
            parts[key] = pd.to_numeric(df[col], errors="coerce")
    if not all(k in parts for k in ("year", "month", "day", "hour")):
        return None
    frame = {"year": parts["year"], "month": parts["month"], "day": parts["day"],
             "hour": parts["hour"], "minute": parts.get("minute", 0)}
    return pd.to_datetime(pd.DataFrame(frame), utc=True, errors="coerce")


def _load_one_weather(path, src_units, col_overrides=None):
    df = pd.read_csv(path)
    cand = dict(C.WEATHER_COLUMN_CANDIDATES)
    if col_overrides:
        cand.update(col_overrides)
    cols = {}
    for key in ("temp", "pressure", "humidity"):
        col = _resolve_col(df, cand[key])
        if col is None:
            raise KeyError(f"Could not find a '{key}' column in {path}. Columns: {list(df.columns)}.")
        cols[key] = col
    dt_col = _resolve_col(df, cand["datetime"])
    dt = parse_datetime_series(df[dt_col]) if dt_col is not None else _assemble_datetime(df)
    if dt is None:
        raise KeyError(f"No datetime in {path}. Columns: {list(df.columns)}.")
    out = pd.DataFrame({
        "datetime": dt,
        "temp": pd.to_numeric(df[cols["temp"]], errors="coerce"),
        "pressure": pd.to_numeric(df[cols["pressure"]], errors="coerce"),
        "humidity": pd.to_numeric(df[cols["humidity"]], errors="coerce"),
    })
    out = out.dropna(subset=["datetime"]).sort_values("datetime")
    for ch in C.CHANNELS:
        v = out[ch].to_numpy(dtype=float, copy=True)
        v[np.isin(v, np.asarray(C.SENTINEL_VALUES, dtype=float))] = np.nan
        v = units.to_canonical(ch, v, src_units[ch])
        lo, hi = C.PLAUSIBLE_RANGE[ch]
        v[(v < lo) | (v > hi)] = np.nan
        out[ch] = v
    return out


def load_hourly_weather(weather):
    if isinstance(weather, str):
        sources = [{"path": weather, "units": C.TRAIN_UNITS, "columns": None}]
    else:
        sources = list(weather)
    frames, used = [], []
    for i, s in enumerate(sources):
        p = s["path"]
        if not os.path.exists(p):
            if s.get("optional"):
                print(f"[weather] source {p} absent; skipping (optional)"); continue
            raise FileNotFoundError(f"weather source not found: {p}")
        f = _load_one_weather(p, s["units"], s.get("columns"))
        f["__src"] = i
        frames.append(f); used.append((p, len(f), f["datetime"].min(), f["datetime"].max()))
    if not frames:
        raise RuntimeError("no weather sources available to load")
    merged = pd.concat(frames, ignore_index=True)
    merged = (merged.sort_values(["datetime", "__src"])
                    .drop_duplicates(subset=["datetime"], keep="first")
                    .drop(columns="__src").set_index("datetime"))
    for p, n, lo, hi in used:
        print(f"[weather] {n:,} rows from {os.path.basename(p)}  ({lo} .. {hi})")
    raw = merged[C.CHANNELS]
    raw_5min = raw.resample("5min").mean()
    hourly = raw_5min.resample("1h", origin="end").mean()
    hourly = hourly.interpolate(method="time", limit=C.MAX_GAP_HOURS, limit_area="inside")
    print(f"[weather] merged hourly grid: {len(hourly):,} hours "
          f"({hourly.index.min()} .. {hourly.index.max()}); 5-min blocks: {len(raw_5min):,}")
          
    return hourly, raw_5min


def load_storm_events(path, use_sbert=True, cz_name_contains=None, state_contains=None):
    cz_pat = cz_name_contains if cz_name_contains is not None else C.CZ_NAME_CONTAINS
    st_pat = state_contains if state_contains is not None else C.STATE_CONTAINS
    wanted = {"cz_name", "state", "event_type", "begin_date_time", "end_date_time", "event_narrative"}
    df = pd.read_csv(path, low_memory=False, usecols=lambda c: c.lower() in wanted)
    colmap = {c.lower(): c for c in df.columns}
    def need(name):
        if name not in colmap:
            raise KeyError(f"Storm Events missing '{name}'. Have: {list(df.columns)}")
        return colmap[name]
    cz = need("cz_name"); st = need("state"); et = need("event_type")
    b = need("begin_date_time"); e = need("end_date_time")
    mask = (df[cz].astype(str).str.lower().str.contains(cz_pat) &
            df[st].astype(str).str.lower().str.contains(st_pat))
    sub = df.loc[mask].copy()
    et_lower = sub[et].astype(str).str.strip().str.lower()
    ignored_mask = et_lower.isin(C.IGNORE_EVENT_TYPES)
    n_ignored = int(ignored_mask.sum())
    ignored_counts = sub.loc[ignored_mask, et].astype(str).value_counts().to_dict()
    sub = sub.loc[~ignored_mask].copy()
    texts = sub[et].astype(str)
    classes, scores, method = map_events_to_classes(list(texts), use_sbert=use_sbert)
    classes = np.asarray(classes, dtype=object); scores = np.asarray(scores, dtype=float)
    n_low = 0
    if C.EVENT_MATCH_MIN_SCORE > 0:
        low = scores < C.EVENT_MATCH_MIN_SCORE
        n_low = int(low.sum()); classes[low] = C.NONE_LABEL
    start = parse_datetime_series(sub[b]); end = parse_datetime_series(sub[e])
    events = pd.DataFrame({"start": start.values, "end": end.values,
                           "event_type": sub[et].astype(str).values,
                           "cls": classes, "match_score": scores}).dropna(subset=["start", "end"])
    events = events[events["end"] >= events["start"]].reset_index(drop=True)
    events = events[events["cls"] != C.NONE_LABEL].reset_index(drop=True)
    prov = (events.groupby(["event_type", "cls"]).size().reset_index(name="n").sort_values("n", ascending=False))
    provenance = {"method": method, "n_events": int(len(events)), "n_ignored": n_ignored,
                  "ignored_counts": ignored_counts, "n_low_confidence_to_none": n_low,
                  "mapping": prov.to_dict(orient="records")}
    print(f"[events] {len(events)} {cz_pat}/{st_pat} events kept; {n_ignored} ignored, {n_low} low->none; via {method}")
    return events, provenance


def load_watches(path):
    if not path or not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    lut = {c.lower(): c for c in df.columns}
    sc = next((lut[k] for k in ("issued", "issue", "stiss", "begin", "starttime") if k in lut), None)
    ec = next((lut[k] for k in ("expired", "expire", "stexp", "end", "endtime") if k in lut), None)
    if sc is None or ec is None:
        print(f"[watches] no issue/expire cols in {list(df.columns)}; skipping"); return None
    tcol = lut.get("type") or lut.get("typetext") or lut.get("phenomena")
    out = pd.DataFrame({
        "start": parse_datetime_series(df[sc]), "end": parse_datetime_series(df[ec]),
        "wtype": (df[tcol].astype(str) if tcol is not None
                  else pd.Series([""] * len(df), index=df.index)),
    }).dropna(subset=["start", "end"])
    # SPC watch TYPE -> class for the y_wide (experiment-0) labels. SVR (severe
    # thunderstorm watch) -> t-storm AND wind, like the SV warning; TOR -> tornado.
    # Watches are convective only (SVR/TOR), so they widen only those classes.
    out["cls"] = out["wtype"].map(_watch_type_to_class_idx)
    print(f"[watches] {len(out)} watch intervals ({out['start'].min()} .. {out['start'].max()})")
    return out


WATCH_TYPE_TO_CLASSES = {"TOR": ("tornado",), "SVR": ("t-storm", "wind")}


def _watch_type_to_class_idx(wtype):
    names = WATCH_TYPE_TO_CLASSES.get(str(wtype).upper().strip(), ())
    return tuple(C.ALL_LABELS.index(n) for n in names if n in C.ALL_LABELS)


WARN_UGC_KEEP = ("ILC031",)
WARN_PHEN_TO_CLASSES = {
    "TO": ("tornado",), "SV": ("t-storm", "wind"),
    "FF": ("flood",), "FA": ("flood",), "FL": ("flood",),
}


def _warn_phen_to_class_idx(phen):
    names = WARN_PHEN_TO_CLASSES.get(str(phen).upper().strip(), ())
    return tuple(C.ALL_LABELS.index(n) for n in names if n in C.ALL_LABELS)


def load_warnings(path, ugc_keep=None):
    ugc_keep = ugc_keep if ugc_keep is not None else WARN_UGC_KEEP
    if not path or not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    lut = {c.lower(): c for c in df.columns}
    sc = next((lut[k] for k in ("utc_issue", "issued", "issue", "stiss",
                                "utc_polygon_begin", "begin", "starttime") if k in lut), None)
    ec = next((lut[k] for k in ("utc_expire", "expired", "expire", "stexp",
                                "utc_polygon_end", "end", "endtime") if k in lut), None)
    if sc is None or ec is None:
        print(f"[warnings] no issue/expire cols in {list(df.columns)}; skipping"); return None
    ugc_col = lut.get("ugc")
    if ugc_col is not None and ugc_keep:
        pat = "|".join(ugc_keep); before = len(df)
        df = df[df[ugc_col].astype(str).str.contains(pat, case=False, na=False)]
        print(f"[warnings] UGC filter {ugc_keep}: kept {len(df):,}/{before:,} rows")
    elif ugc_col is None:
        print(f"[warnings] no 'ugc' column; keeping all zones")
    pcol = lut.get("phenomena") or lut.get("phenom") or lut.get("type")
    out = pd.DataFrame({
        "start": parse_datetime_series(df[sc]),
        "end": parse_datetime_series(df[ec]),
        "phen": (df[pcol].astype(str) if pcol is not None
                 else pd.Series([""] * len(df), index=df.index)),
    }).dropna(subset=["start", "end"])
    out["cls"] = out["phen"].map(_warn_phen_to_class_idx)
    print(f"[warnings] {len(out)} warning intervals ({out['start'].min()} .. {out['start'].max()})")
    return out


def _interp_channel(grid_sec, anc_sec, anc_val, method):
    if anc_sec.size == 0:
        return np.full(grid_sec.shape, np.nan)
    if method == "log" and np.all(anc_val > 0):
        return np.exp(np.interp(grid_sec, anc_sec, np.log(anc_val)))
    return np.interp(grid_sec, anc_sec, anc_val)


def _idx_to_epoch_sec(idx) -> np.ndarray:
    """DatetimeIndex -> int64 epoch SECONDS, independent of the index's time
    resolution (datetime64[ns] vs [us] etc.). The old `asi8 // 1e9` assumed ns;
    a CSV that parses to microsecond resolution would silently produce t_sec that
    is 1000x too small. Going through datetime64[s] truncates correctly for any
    unit (matters now that we ingest several stations' CSVs from varied sources)."""
    return idx.values.astype("datetime64[s]").astype("int64")


def _vals_to_epoch_sec(vals) -> np.ndarray:
    """Same, for a pandas .values datetime64 array (any resolution)."""
    return np.asarray(vals).astype("datetime64[s]").astype("int64")


def build_subhourly(chan_obs, t_end_sec, method=C.INTERP_METHOD):
    step = C.SUBHOURLY_INTERVAL_MIN * 60
    L = C.SUBHOURLY_WINDOW_LEN
    grid = t_end_sec - step * np.arange(L - 1, -1, -1)
    lo = t_end_sec - (C.SUBHOURLY_WINDOW_HOURS + 1) * 3600
    out = np.empty((L, C.N_CHANNELS), dtype=np.float64)
    for ci, (sec, val) in enumerate(chan_obs):
        i0 = np.searchsorted(sec, lo, side="left")
        i1 = np.searchsorted(sec, t_end_sec, side="right")
        out[:, ci] = _interp_channel(grid, sec[i0:i1], val[i0:i1], method)
    return out


def _overlap_labels(t_sec, horizon_h, ev_start, ev_end, ev_cls, future_temp_c=None):
    he = t_sec + horizon_h * 3600
    m = (ev_start <= he) & (ev_end >= t_sec)
    vec = np.zeros(C.N_OUTPUTS, dtype=np.float32)
    if m.any():
        for ci in np.unique(ev_cls[m]):
            vec[ci] = 1.0
    if future_temp_c is not None and future_temp_c.size:
        if _HEAT_THR_C is not None and np.any(future_temp_c > _HEAT_THR_C):
            vec[_HEAT_IDX] = 1.0
        if _COLD_THR_C is not None and np.any(future_temp_c < _COLD_THR_C):
            vec[_COLD_IDX] = 1.0
    if vec[:len(C.CLASSES)].sum() == 0:
        vec[C.ALL_LABELS.index(C.NONE_LABEL)] = 1.0
    return vec


def _temp_signal(future_temp_c):
    if future_temp_c is None or not future_temp_c.size:
        return False
    hot = _HEAT_THR_C is not None and bool(np.any(future_temp_c > _HEAT_THR_C))
    cold = _COLD_THR_C is not None and bool(np.any(future_temp_c < _COLD_THR_C))
    return hot or cold


def _any_overlap(t_sec, horizon_h, s, e, return_count=False):
    if s.size == 0:
        return 0 if return_count else False
    he = t_sec + horizon_h * 3600
    mask = (s <= he) & (e >= t_sec)
    return int(mask.sum()) if return_count else bool(mask.any())


def _count_overlap(t_sec, horizon_h, s, e):
    if s.size == 0:
        return 0
    he = t_sec + horizon_h * 3600
    return int(((s <= he) & (e >= t_sec)).sum())


def print_terminal_histogram(weights, width=80, height=30):
    data_min, data_max = weights.min(), weights.max()
    data_mean = weights.mean(); data_median = np.median(weights)
    counts, _ = np.histogram(weights, bins=width, range=(data_min, data_max))
    max_count = counts.max()
    if max_count == 0:
        print("empty"); return
    scaled = np.round((counts / max_count) * height).astype(int)
    for r in range(height - 1, -1, -1):
        print("".join("*" if scaled[c] > r else " " for c in range(width)))
    print("-" * width)
    print(f"mean {data_mean:.2f}  median {data_median:.2f}  min {data_min:.2f}  max {data_max:.2f}")


def build_dataset(weather_path=None, storm_events_path=None, watches_path=None,
                  warnings_path=None, use_sbert=True, out_path=None,
                  region=None, return_arrays=False):
    # region (optional): dict overriding the Cook defaults for a different station.
    #   {"cz_name_contains": "champaign", "state_contains": "illinois",
    #    "warn_ugc_keep": ("ILC019",)}.  None -> use config (Cook County).
    region = region or {}
    cz_name = region.get("cz_name_contains")
    state_name = region.get("state_contains")
    ugc_keep = region.get("warn_ugc_keep")
    out_path = out_path or os.path.join(C.BUILD_DIR, "dataset.npz")
    weather = weather_path if weather_path is not None else C.WEATHER_SOURCES
    hourly, raw = load_hourly_weather(weather)
    events, provenance = load_storm_events(storm_events_path, use_sbert=use_sbert,
                                           cz_name_contains=cz_name, state_contains=state_name)
    watches = load_watches(watches_path)
    if warnings_path is None:
        _wp = os.path.join(C.DATA_DIR, "spc_warnings_raw.csv")
        warnings_path = _wp if os.path.exists(_wp) else None
    warnings = load_warnings(warnings_path, ugc_keep=ugc_keep)

    idx = hourly.index
    if len(idx) < C.HOURLY_WINDOW_LEN + C.HORIZON_24H + 1:
        raise RuntimeError("not enough hourly data")
    anchor_vals = hourly[C.CHANNELS].to_numpy(dtype=np.float64)
    hourly_sec = _idx_to_epoch_sec(idx)
    raw_sec_all = _idx_to_epoch_sec(raw.index)
    chan_obs = []
    for ch in C.CHANNELS:
        v = raw[ch].to_numpy(dtype=np.float64); good = ~np.isnan(v)
        chan_obs.append((raw_sec_all[good], v[good]))

    ev_start = _vals_to_epoch_sec(events["start"].values)
    ev_end = _vals_to_epoch_sec(events["end"].values)
    ev_cls = events["cls"].map(lambda c: C.ALL_LABELS.index(c)).to_numpy()
    tol = int(C.LABEL_TOLERANCE_MIN) * 60
    if tol:
        ev_start = ev_start - tol; ev_end = ev_end + tol
    if watches is not None and len(watches):
        wa_start = _vals_to_epoch_sec(watches["start"].values)
        wa_end = _vals_to_epoch_sec(watches["end"].values)
    else:
        wa_start = wa_end = np.array([], dtype=np.int64)
    if warnings is not None and len(warnings):
        warn_start = _vals_to_epoch_sec(warnings["start"].values)
        warn_end = _vals_to_epoch_sec(warnings["end"].values)
    else:
        warn_start = warn_end = np.array([], dtype=np.int64)

    # z source: events UNION warnings (test-only)
    zw_s, zw_e, zw_c = [], [], []
    if warnings is not None and len(warnings):
        _ws = _vals_to_epoch_sec(warnings["start"].values)
        _we = _vals_to_epoch_sec(warnings["end"].values)
        for s_i, e_i, cls_i in zip(_ws, _we, warnings["cls"].values):
            for ci in cls_i:
                zw_s.append(int(s_i)); zw_e.append(int(e_i)); zw_c.append(int(ci))
    zw_s = np.asarray(zw_s, dtype=np.int64); zw_e = np.asarray(zw_e, dtype=np.int64)
    zw_c = np.asarray(zw_c, dtype=np.int64)
    if tol and zw_s.size:
        zw_s = zw_s - tol; zw_e = zw_e + tol
    z_start = np.concatenate([ev_start, zw_s]) if zw_s.size else ev_start
    z_end = np.concatenate([ev_end, zw_e]) if zw_e.size else ev_end
    z_cls = np.concatenate([ev_cls, zw_c]) if zw_c.size else ev_cls
    print(f"[dataset] z-source: {len(ev_cls):,} events + {zw_s.size:,} warning-class rows = {z_cls.size:,} total")

    # y_wide source: z-source UNION watches (experiment 0; a watch makes its
    # class positive). Watches are SVR/TOR -> widen tornado/t-storm/wind only.
    # Same tolerance widen as events/warnings. Trained ON in experiment 0/0.5;
    # always scored against z (the fixed ruler), never against y_wide.
    ww_s, ww_e, ww_c = [], [], []
    if watches is not None and len(watches) and "cls" in watches:
        _wa_s = _vals_to_epoch_sec(watches["start"].values)
        _wa_e = _vals_to_epoch_sec(watches["end"].values)
        for s_i, e_i, cls_i in zip(_wa_s, _wa_e, watches["cls"].values):
            for ci in cls_i:
                ww_s.append(int(s_i)); ww_e.append(int(e_i)); ww_c.append(int(ci))
    ww_s = np.asarray(ww_s, dtype=np.int64); ww_e = np.asarray(ww_e, dtype=np.int64)
    ww_c = np.asarray(ww_c, dtype=np.int64)
    if tol and ww_s.size:
        ww_s = ww_s - tol; ww_e = ww_e + tol
    yw_start = np.concatenate([z_start, ww_s]) if ww_s.size else z_start
    yw_end = np.concatenate([z_end, ww_e]) if ww_e.size else z_end
    yw_cls = np.concatenate([z_cls, ww_c]) if ww_c.size else z_cls
    print(f"[dataset] y_wide-source: z-source {z_cls.size:,} + {ww_s.size:,} watch-class "
          f"rows = {yw_cls.size:,} total")

    cutoff_pos = int(len(idx) * TRAIN_FRAC)
    train_vals = anchor_vals[:cutoff_pos]
    mu = np.nanmean(train_vals, axis=0); sd = np.nanstd(train_vals, axis=0); sd[sd == 0] = 1.0

    first_pos = C.HOURLY_WINDOW_LEN - 1
    last_pos = len(idx) - 1 - C.HORIZON_24H
    positions = range(first_pos, last_pos + 1, C.WINDOW_STRIDE_HOURS)

    X, X2, DOH, TS, Y1, Y24, W, W1, Z1, Z24, YW1, YW24 = ([] for _ in range(12))
    skipped = 0
    for pos in positions:
        t_sec = int(hourly_sec[pos])
        win = anchor_vals[pos - C.HOURLY_WINDOW_LEN + 1: pos + 1]
        if win.shape[0] != C.HOURLY_WINDOW_LEN:
            skipped += 1; continue
        if np.isnan(win).sum() / win.size > C.MAX_MISSING_FRACTION:
            skipped += 1; continue
        xi = win.copy()
        if np.isnan(xi).any():
            for ci in range(C.N_CHANNELS):
                col = xi[:, ci]; nans = np.isnan(col)
                if nans.any():
                    col[nans] = (np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), col[~nans])
                                 if (~nans).any() else 0.0)
                xi[:, ci] = col
        x2i = build_subhourly(chan_obs, t_sec)
        if np.isnan(x2i).any():
            skipped += 1; continue
        ts = pd.Timestamp(t_sec, unit="s", tz="UTC")
        doh = np.zeros(C.DATE_FEAT_DIM, dtype=np.float32); doh[ts.month - 1] = 1.0
        fut1 = anchor_vals[pos + 1: pos + 1 + C.HORIZON_1H, 0]
        fut24 = anchor_vals[pos + 1: pos + 1 + C.HORIZON_24H, 0]
        y1 = _overlap_labels(t_sec, C.HORIZON_1H, ev_start, ev_end, ev_cls, fut1)
        y24 = _overlap_labels(t_sec, C.HORIZON_24H, ev_start, ev_end, ev_cls, fut24)
        z1 = _overlap_labels(t_sec, C.HORIZON_1H, z_start, z_end, z_cls, fut1)
        z24 = _overlap_labels(t_sec, C.HORIZON_24H, z_start, z_end, z_cls, fut24)
        yw1 = _overlap_labels(t_sec, C.HORIZON_1H, yw_start, yw_end, yw_cls, fut1)
        yw24 = _overlap_labels(t_sec, C.HORIZON_24H, yw_start, yw_end, yw_cls, fut24)

        watch_count_24 = _any_overlap(t_sec, C.HORIZON_24H, wa_start, wa_end, return_count=True)
        warn_count_24 = _any_overlap(t_sec, C.HORIZON_24H, warn_start, warn_end, return_count=True)
        event_count_24 = min(3, _any_overlap(t_sec, C.HORIZON_24H, ev_start, ev_end, return_count=True))
        watch_count_1 = _any_overlap(t_sec, C.HORIZON_1H, wa_start, wa_end, return_count=True)
        warn_count_1 = _any_overlap(t_sec, C.HORIZON_1H, warn_start, warn_end, return_count=True)
        event_count_1 = min(3, _any_overlap(t_sec, C.HORIZON_1H, ev_start, ev_end, return_count=True))
        w = (C.WEIGHT_BASE
             + C.WEIGHT_WATCH_ONLY * math.log1p(watch_count_24)
             + C.WEIGHT_WARNING * math.log1p(warn_count_24)
             + C.WEIGHT_STORM_EVENT * math.log1p(event_count_24)
             + C.WEIGHT_WATCH_ONLY * float(_temp_signal(fut24)))
        w1 = (C.WEIGHT_BASE
              + C.WEIGHT_WATCH_ONLY * math.log1p(watch_count_1)
              + C.WEIGHT_WARNING * math.log1p(warn_count_1)
              + C.WEIGHT_STORM_EVENT * math.log1p(event_count_1)
              + C.WEIGHT_WATCH_ONLY * float(_temp_signal(fut1)))

        X.append(xi.astype(np.float32)); X2.append(x2i.astype(np.float32))
        DOH.append(doh); TS.append(t_sec)
        Y1.append(y1); Y24.append(y24); W.append(w); W1.append(w1)
        Z1.append(z1); Z24.append(z24)
        YW1.append(yw1); YW24.append(yw24)

    X = np.stack(X); X2 = np.stack(X2); DOH = np.stack(DOH)
    TS = np.asarray(TS, dtype=np.int64)
    Y1 = np.stack(Y1); Y24 = np.stack(Y24)
    W = np.asarray(W, dtype=np.float32); W1 = np.asarray(W1, dtype=np.float32)
    Z1 = np.stack(Z1); Z24 = np.stack(Z24)
    YW1 = np.stack(YW1); YW24 = np.stack(YW24)
    print(f"[dataset] built {len(X):,} samples (skipped {skipped:,})")
    print(f"weight: mean W={np.mean(W):.4f}  mean W1={np.mean(W1):.4f}")
    print("[dataset] convective positive rates (24h):  y -> z -> y_wide")
    for j, lab in enumerate(C.ALL_LABELS):
        if lab in ("tornado", "flood", "wind", "t-storm"):
            print(f"           {lab:9s}: {Y24[:,j].mean():.4f} -> {Z24[:,j].mean():.4f} "
                  f"-> {YW24[:,j].mean():.4f}")
    arrays = dict(
        X=X, X2=X2, date_oh=DOH, t_sec=TS, y_1h=Y1, y_24h=Y24, weight=W,
        scaler_mean=mu.astype(np.float32), scaler_std=sd.astype(np.float32),
        labels=np.array(C.ALL_LABELS), weight_hour=W1, z_1h=Z1, z_24h=Z24,
        y_wide_1h=YW1, y_wide_24h=YW24)
    if return_arrays:
        # the multi-station pooler collects these in-memory instead of saving
        return arrays
    np.savez_compressed(out_path, **arrays)
    with open(os.path.join(C.BUILD_DIR, "event_class_map.json"), "w") as f:
        json.dump(provenance, f, indent=2)
    print(f"[dataset] saved -> {out_path}")
    return out_path


# ===========================================================================
# Multi-station pooling (Cook + Champaign + Will/Joliet + ... )
# ===========================================================================
# Each station is a dict:
#   {"name": "cook",
#    "weather_path": ".../asos_ord.csv"  (or a WEATHER_SOURCES-style list),
#    "region": {"cz_name_contains": "cook", "state_contains": "illinois",
#               "warn_ugc_keep": ("ILC031",)},
#    "storm_events_path": ".../storm_events_IL.csv"  (statewide; filtered by region),
#    "watches_path": ".../watches.csv" or None,
#    "warnings_path": ".../warnings_<wfo>.csv" or None}
#
# The TEST split is taken from ONE station only (default "cook") so the ruler is
# identical to every prior Cook-only experiment — train/val pool all stations,
# but we always score Cook-only z. The split is computed by TIME with a real
# embargo gap applied across the POOLED timeline, so the same storm (same t_sec
# in two nearby cities) can never straddle the train/test boundary (spatial-
# leakage guard). Per-station scaling (default) z-scores each station by its own
# train-period stats so a pattern is "anomaly vs local normal" and transfers
# across locations; set per_station_scaling=False for one shared scaler.

def _fit_scaler_from_windows(X_tr):
    """Per-channel mean/std over (samples, time) of the TRAIN windows (N,L,3).
    dtype=float64 is REQUIRED: X is float32 and N*L is ~30M, so a multi-axis
    reduction over the strided channel-last array drops out of numpy's pairwise
    summation into naive float32 accumulation and loses precision badly (the
    pressure channel, ~1016, came out ~676). float64 accumulation fixes it."""
    mu = X_tr.mean(axis=(0, 1), dtype=np.float64)
    sd = X_tr.std(axis=(0, 1), dtype=np.float64); sd[sd == 0] = 1.0
    return mu.astype(np.float32), sd.astype(np.float32)


def build_multistation(stations, out_path=None, test_station="cook",
                       per_station_scaling=True, use_sbert=True):
    out_path = out_path or os.path.join(C.BUILD_DIR, "dataset_multi.npz")
    embargo_sec = int(C.SPLIT_EMBARGO_HOURS) * 3600

    # ---- build each station's raw arrays (unnormalized windows) -------------
    per = []
    for si, st in enumerate(stations):
        print(f"\n========== station {si}: {st['name']} ==========")
        a = build_dataset(weather_path=st["weather_path"],
                          storm_events_path=st["storm_events_path"],
                          watches_path=st.get("watches_path"),
                          warnings_path=st.get("warnings_path"),
                          use_sbert=use_sbert, region=st.get("region"),
                          return_arrays=True)
        a["station"] = np.full(len(a["t_sec"]), si, dtype=np.int64)
        a["__name"] = st["name"]
        per.append(a)

    test_si = next(i for i, st in enumerate(stations) if st["name"] == test_station)

    # ---- time boundaries from the TEST station's timeline (so the Cook test
    #      window range matches prior experiments) --------------------------
    t_test = np.sort(per[test_si]["t_sec"])
    n = len(t_test)
    t_tr_end = t_test[int(n * C.TRAIN_SPLIT)]                 # train/val boundary time
    t_va_end = t_test[int(n * (C.TRAIN_SPLIT + C.VAL_SPLIT))]  # val/test boundary time
    print(f"\n[multi] time boundaries (from {test_station}): "
          f"train< {pd.Timestamp(t_tr_end, unit='s', tz='UTC')}  "
          f"test>= {pd.Timestamp(t_va_end, unit='s', tz='UTC')}  embargo={C.SPLIT_EMBARGO_HOURS}h")

    # ---- per-station scaling (optional) then pool ---------------------------
    keys = ["X", "X2", "date_oh", "t_sec", "y_1h", "y_24h", "weight", "weight_hour",
            "z_1h", "z_24h", "y_wide_1h", "y_wide_24h", "station"]
    station_scalers = {}        # name -> (mu, sd); the REAL per-station stats, kept
                                # so live inference can normalize a single station
                                # (the pooled npz stores an identity scaler).
    for a in per:
        if per_station_scaling:
            tr_mask = a["t_sec"] < (t_tr_end - embargo_sec)
            if tr_mask.sum() < 10:
                raise RuntimeError(f"station {a['__name']}: too few train windows to fit scaler")
            mu, sd = _fit_scaler_from_windows(a["X"][tr_mask])
            station_scalers[a["__name"]] = (mu, sd)
            a["X"] = ((a["X"] - mu) / sd).astype(np.float32)
            a["X2"] = ((a["X2"] - mu) / sd).astype(np.float32)
            print(f"[multi] {a['__name']}: per-station scaler mu={np.round(mu,2)} sd={np.round(sd,2)}")
    pooled = {k: np.concatenate([a[k] for a in per], axis=0) for k in keys}

    # If per-station scaling already applied, store an IDENTITY scaler so the
    # training code's normalize() is a no-op (data is already normalized). If
    # shared scaling, fit one scaler on the pooled TRAIN windows.
    if per_station_scaling:
        mu_all = np.zeros(C.N_CHANNELS, np.float32); sd_all = np.ones(C.N_CHANNELS, np.float32)
    else:
        tr_mask = pooled["t_sec"] < (t_tr_end - embargo_sec)
        mu_all, sd_all = _fit_scaler_from_windows(pooled["X"][tr_mask])
    pooled["scaler_mean"] = mu_all; pooled["scaler_std"] = sd_all
    pooled["labels"] = np.array(C.ALL_LABELS)

    # ---- cross-station TIME-embargo split; TEST is test_station only --------
    ts = pooled["t_sec"]; stn = pooled["station"]
    tr = np.flatnonzero(ts < (t_tr_end - embargo_sec))
    va = np.flatnonzero((ts >= t_tr_end) & (ts < (t_va_end - embargo_sec)))
    te = np.flatnonzero((ts >= t_va_end) & (stn == test_si))   # Cook-only test
    pooled["split_tr"] = tr.astype(np.int64)
    pooled["split_va"] = va.astype(np.int64)
    pooled["split_te"] = te.astype(np.int64)

    # ---- leakage assertion: no train/test storm-time overlap across stations
    if te.size and tr.size:
        assert ts[tr].max() < ts[te].min(), "TIME LEAK: a train window is at/after a test window"
        gap_h = (ts[te].min() - ts[tr].max()) / 3600.0
        assert gap_h >= C.SPLIT_EMBARGO_HOURS - 1, f"embargo gap only {gap_h:.0f}h < {C.SPLIT_EMBARGO_HOURS}h"
        print(f"[multi] leakage guard OK: train->test gap = {gap_h:.0f}h (>= {C.SPLIT_EMBARGO_HOURS}h embargo)")

    print(f"[multi] pooled {len(ts):,} windows from {len(stations)} stations; "
          f"split tr={tr.size:,} va={va.size:,} te={te.size:,} (test={test_station} only)")
    print(f"[multi] convective positive rates (z_24h, POOLED train):")
    trz = pooled["z_24h"][tr]
    for j, lab in enumerate(C.ALL_LABELS):
        if lab in ("tornado", "flood", "wind", "t-storm"):
            print(f"          {lab:9s}: pooled-train pos rate {trz[:, j].mean():.4f} "
                  f"({int(trz[:, j].sum())} positives)")

    # stash the real per-station scalers in the pooled npz for provenance, and
    # write a standalone scaler_{name}.npz next to it so live_engine can load the
    # target station's mean/std (the pooled scaler is identity under per-station).
    if per_station_scaling and station_scalers:
        out_dir = os.path.dirname(out_path) or "."
        for name, (mu, sd) in station_scalers.items():
            pooled[f"station_scaler_mean_{name}"] = mu.astype(np.float32)
            pooled[f"station_scaler_std_{name}"] = sd.astype(np.float32)
            sp_path = os.path.join(out_dir, f"scaler_{name}.npz")
            np.savez(sp_path, mean=mu.astype(np.float32), std=sd.astype(np.float32))
            print(f"[multi] wrote per-station scaler -> {sp_path}  "
                  f"(mu={np.round(mu,2)} sd={np.round(sd,2)})")

    np.savez_compressed(out_path, **pooled)
    print(f"[multi] saved -> {out_path}")
    return out_path

if __name__ == "__main__":
  
    from datetime import datetime
    from meteostat import Point, Monthly

    # Coordinates for Joliet, Illinois
    lat, lon = 41.525, -88.081
    joliet = Point(lat, lon)

    # Retrieve data for the year (e.g., last fully recorded year)
    start = datetime(2025, 1, 1)
    end = datetime(2025, 12, 31)

    # Fetch monthly data and calculate the annual mean
    data = Monthly(joliet, start, end)
    data_df = data.fetch()

    # Calculate the mean of all monthly mean temperatures
    annual_mean = data_df['tavg'].mean()
    print(f"The mean temperature for 2025 was: {annual_mean:.2f}°C")

# Historical 30-year climate normals for Joliet/Chicago area
    monthly_humidity = {
        'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
        'Rel_Humidity_Percent': [0, 0, 2, 5, 12, 33, 50, 44, 36, 10, 5, 0]
    }

    df = pd.DataFrame(monthly_humidity)

    # Calculate the annual mean relative humidity
    annual_mean = df['Rel_Humidity_Percent'].mean()
    print(f"Annual Mean Relative Humidity: {annual_mean:.2f}%")

    local_df = pd.read_csv('/Users/dominic.muscatella/Downloads/weather_predictor_src/live_work/live_LOCAL.csv')

    # 2. Convert string to datetime using the exact format
    local_df['valid'] = pd.to_datetime(local_df['valid'], format='%Y-%m-%d %H:%M')

    # 3. Filter the range (Pandas handles the time part automatically)
    start_date = '2026-06-11 00:00'
    end_date = '2026-06-18 00:00'

    local_filtered_df = local_df[local_df['valid'].between(start_date, end_date)]
    local_statistics = local_filtered_df[['tmpf', 'relh', 'pressure_hpa']].agg(['mean', 'std'])
    co9_df = pd.read_csv('/Users/dominic.muscatella/Downloads/weather_predictor_src/live_work/live_C09.csv')

    # 2. Convert string to datetime using the exact format
    co9_df['valid'] = pd.to_datetime(co9_df['valid'], format='%Y-%m-%d %H:%M')

    # 3. Filter the range (Pandas handles the time part automatically)

    co9_filtered_df = co9_df[co9_df['valid'].between(start_date, end_date)]
    co9_statistics = co9_filtered_df[['tmpf', 'relh', 'pressure_hpa']].agg(['mean', 'std'])
    
    print("local stats:")
    print(local_statistics.to_string(index=False))
    print("co9 stats:")
    print(co9_statistics.to_string(index=False))

    # if local is higher than remote, we want the scaler mean to be higher
    # if the local is lower than the remote, we want the scaler mean to be lower
    # same with the std deviations
    # so local - remote = positive number if local is higher, negative number if remote is higher
    print("stat diff, local - co9")
    stat_diff = local_statistics - co9_statistics
    print(stat_diff.to_string(index=False))


    base_Stats = {"mu":{ # collected from joliet station
                    "tmpf":71.8278,
                    "baro":1016.6022,
                    "rel_h": 10.8397
    },
                "std":{
                    "tmpf":18.419,
                    "baro":7.2405,
                    "rel_h": 11.5468
                }}  
    
    print("original, joliet derived scalers:")
    print(json.dumps(base_Stats, indent=4))
    print()
    base_Stats["std"]["baro"] = base_Stats["std"]["baro"] +float(stat_diff.at['std', 'pressure_hpa'])
    # base_Stats["std"]["tmpf"] = base_Stats["std"]["tmpf"] +float(stat_diff.at['std', 'tmpf'])
    # base_Stats["std"]["rel_h"] = base_Stats["std"]["rel_h"] +float(stat_diff.at['std', 'relh'])

    base_Stats["mu"]["baro"] = base_Stats["mu"]["baro"] +float(stat_diff.at['mean', 'pressure_hpa'])
    # base_Stats["mu"]["tmpf"] = base_Stats["mu"]["tmpf"] +float(stat_diff.at['mean', 'tmpf'])
    # base_Stats["mu"]["rel_h"] = base_Stats["mu"]["rel_h"] +float(stat_diff.at['mean', 'relh'])
    print("modified stats:")
    print(json.dumps(base_Stats, indent=4))
    print()
    out_dir = os.path.dirname("/Users/dominic.muscatella/Downloads/weather_predictor_src/") or "."
    name = "LOCAL"
    mu = np.array([base_Stats["mu"]["rel_h"], base_Stats["mu"]["baro"], base_Stats["mu"]["tmpf"]])
    sd = np.array([base_Stats["std"]["rel_h"], base_Stats["std"]["baro"], base_Stats["std"]["tmpf"]])
    sp_path = os.path.join(out_dir, f"scaler_{name}.npz")
    np.savez(sp_path, mean=mu.astype(np.float32), std=sd.astype(np.float32))
    print(f"[multi] wrote per-station scaler -> {sp_path}  "
            f"(mu={np.round(mu,2)} sd={np.round(sd,2)})")