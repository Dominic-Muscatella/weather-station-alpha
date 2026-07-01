import os
import json
from datetime import datetime, timedelta, timezone
import numpy as np
from engine.ensemble.knn import (knn_live,
                                 load_knn_refs,
                                 append_embedding,
                                 embed)

from engine.ensemble.monte_carlo import mc_predict
import config as C

from engine.model import load_model
from system_management.sys_reporting import (write_to_start_lock_file,
                                  write_to_knn_lock_file,
                                  write_to_monte_lock_file)
from engine.ASOS_data_fetch import (fetch_asos_range,
                                    fetch_recent_asos,
                                    fetch_active_warnings,
                                    fetch_warning_intervals,
                                    warnings_active_at,
                                    write_live_csv)
from engine.data_handling import (assemble_windows,
                                  _load_grid,
                                  _slice_window)


def load_scaler(model_dir, scaler_npz):
    def _read(z):
        if "mean" in z and "std" in z:
            return z["mean"], z["std"]
        if "scaler_mean" in z and "scaler_std" in z:
            return z["scaler_mean"], z["scaler_std"]
        raise KeyError(f"no mean/std or scaler_mean/scaler_std in archive; "
                       f"keys present: {list(z.keys())}")
    p = os.path.join(model_dir, "scaler.npz")
    if os.path.exists(p):
        return _read(np.load(p))
    if scaler_npz and os.path.exists(scaler_npz):
        return _read(np.load(scaler_npz, allow_pickle=True))
    raise FileNotFoundError(
        "no scaler found: put scaler.npz {mean,std} in the model dir, or pass "
        "--scaler-npz pointing at a per-station scaler (build/scaler_<name>.npz) "
        "or the build npz. NOTE: multi-station builds store an IDENTITY pooled "
        "scaler; for live use you must supply the TARGET STATION's real mean/std.")


def _week_start(dt_obj):
    d = dt_obj - timedelta(days=dt_obj.weekday())
    return d.replace(hour=0, minute=0, second=0, microsecond=0)


def rotate_log(history, log_path, station, now):
    cutoff = now - timedelta(days=C.ROLLING_DAYS)
    keep, drop = [], []
    for r in history:
        ts = r.get("data_latest_utc") or r.get("run_utc")
        try:
            t = datetime.fromisoformat(ts)
        except Exception:
            keep.append(r); continue
        (drop if t < cutoff else keep).append(r)
    if drop:
        arc_dir = os.path.join(os.path.dirname(log_path) or ".", "archive")
        os.makedirs(arc_dir, exist_ok=True)

        buckets = {}
        for r in drop:
            ts = r.get("data_latest_utc") or r.get("run_utc")
            try:
                wk = _week_start(datetime.fromisoformat(ts))
            except Exception:
                wk = _week_start(now)
            buckets.setdefault(wk.strftime("%Y%m%d"), []).append(r)
        for wk, recs in buckets.items():
            ap = os.path.join(arc_dir, f"live_log_{station}_{wk}.json")
            prior = []
            if os.path.exists(ap):
                try:
                    with open(ap) as f:
                        prior = json.load(f)
                except Exception:
                    prior = []
            prior.extend(recs)
            tmp = ap + ".tmp"
            with open(tmp, "w") as f:
                json.dump(prior, f)
            os.replace(tmp, ap)
        print(f"[live] archived {len(drop)} record(s) older than {C.ROLLING_DAYS}d "
              f"into {arc_dir}/ ({len(buckets)} week file(s))")
    return keep


def _build_record(args, xh, xs, doh, quality, models, scaler, device,
                  now, last_fetch, fetch_start, active_warn, backfill=False):
    labels = list(C.ALL_LABELS); n_out = len(labels)
    m1, m24 = models; mean, std = scaler
    anchor_obs = {ch: float(xh[-1, ci]) for ci, ch in enumerate(C.CHANNELS)}
    mean1, lo1, hi1 = mc_predict(m1, xh, xs, doh, mean, std, device)
    mean24, lo24, hi24 = mc_predict(m24, xh, xs, doh, mean, std, device)
    write_to_monte_lock_file()
    print("[model run] embedding for knn...")
    qE = embed(m1, xh, xs, doh, mean, std, device)
    print("[model run] generating record...")
    record = {
        "run_utc": now.isoformat(),
        "last_fetch_utc": last_fetch.isoformat(),
        "data_latest_utc": quality["anchor_utc"],
        "fetch_seconds": round((last_fetch - fetch_start).total_seconds(), 1),
        "station": args.station, "location": args.location, "arch": args.arch,
        "quality": quality, "obs_latest": anchor_obs,
        "model_1h": {lab: {"prob": float(mean1[i]), "lo": float(lo1[i]), "hi": float(hi1[i])}
                     for i, lab in enumerate(labels)},
        "model_24h": {lab: {"prob": float(mean24[i]), "lo": float(lo24[i]), "hi": float(hi24[i])}
                      for i, lab in enumerate(labels)},
        "active_warnings": active_warn,
    }
    if backfill:
        record["backfill"] = True
    if args.knn_csv and os.path.exists(args.knn_csv):
        print("[model run] attempting knn...")
        ref_E, ref_cls, priors = load_knn_refs(args.knn_csv, n_out)
        knn = {}
        for k in [int(x) for x in args.knn_ks.split(",")]:
            v1, conf = knn_live(qE, ref_E, ref_cls, n_out, k, prior=None)
            v2, _ = knn_live(qE, ref_E, ref_cls, n_out, k, prior=priors)
            knn[f"k{k}"] = {
                "v1_distance_weighted": {lab: float(v1[i]) for i, lab in enumerate(labels)},
                "v1_confidence": conf,
                "v2_prior_reweighted": {lab: float(v2[i]) for i, lab in enumerate(labels)},
            }
        record["knn"] = knn
        write_to_knn_lock_file()
    return record, qE


def backfill_cold_start(args, device):
    now = datetime.now(timezone.utc)
    fetch_start = datetime.now(timezone.utc)
    span_days = args.backfill_days + 8
    start = now - timedelta(days=span_days)
    print(f"[backfill] cold start: fetching {span_days}d of {args.station} "
          f"to replay last {args.backfill_days}d hourly...")

    last_fetch = datetime.now(timezone.utc)
    if args.station != "LOCAL":
        df = fetch_asos_range(args.station, start, now)
        csv_path = os.path.join(args.work_dir, f"live_{args.station}.csv")
        write_live_csv(df, csv_path)
    else:
        csv_path = os.path.join("/mnt/DeepData", f"live_{args.station}.csv")
    print("using csv:", csv_path)
    grid = _load_grid(csv_path)
    n_hourly, idx_sec = grid["n"], grid["idx_sec"]
    if n_hourly < C.HOURLY_WINDOW_LEN + 1:
        print(f"[backfill] only {n_hourly} hourly steps (<{C.HOURLY_WINDOW_LEN}); "
              f"skipping backfill, going straight to live.")
        return []

    cutoff_sec = (now - timedelta(days=args.backfill_days)).timestamp()
    first_pos = C.HOURLY_WINDOW_LEN - 1
    positions = [p for p in range(first_pos, n_hourly) if idx_sec[p] >= cutoff_sec]
    if not positions:
        print("[backfill] no anchor positions in window; going live.")
        return []

    warn_intervals = []
    if args.warn_wfo:
        warn_intervals = fetch_warning_intervals(args.warn_wfo, args.warn_ugc,
                                                 start, now)
    scaler = load_scaler(args.model, args.scaler_npz)
    models = (load_model(args.arch, os.path.join(args.model, "model_1h.pt"), device),
              load_model(args.arch, os.path.join(args.model, "model_24h.pt"), device))

    history = []
    print(f"[backfill] replaying {len(positions)} hourly anchors...")
    for n, p in enumerate(positions):
        try:
            xh, xs, doh, quality = _slice_window(grid, anchor_pos=p)
        except RuntimeError:
            continue
        anchor_dt = datetime.fromisoformat(quality["anchor_utc"])
        active = warnings_active_at(warn_intervals, anchor_dt)
        rec, qE = _build_record(args, xh, xs, doh, quality, models, scaler, device,
                                now=anchor_dt, last_fetch=last_fetch,
                                fetch_start=fetch_start, active_warn=active,
                                backfill=True)
        append_embedding(args.embeddings, args.station, rec["data_latest_utc"], qE)
        history.append(rec)
        if (n + 1) % 24 == 0:
            print(f"[backfill]   {n+1}/{len(positions)} ({rec['data_latest_utc']})")
    tmp = args.out + ".tmp"
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(tmp, "w") as f:
        json.dump(history, f, indent=2)
    os.replace(tmp, args.out)
    print(f"[backfill] wrote {len(history)} reconstructed record(s) to {args.out}")
    return history


def run_cycle(args, device):
    write_to_start_lock_file()
    now = datetime.now(timezone.utc)
    write_to_monte_lock_file()
    fetch_start = datetime.now(timezone.utc)
    if args.station != "LOCAL":

        csv_path = os.path.join(args.work_dir, f"live_{args.station}.csv")

    else:
        csv_path = os.path.join("/mnt/DeepData", f"live_{args.station}.csv")
    print("using csv:", csv_path)

    last_fetch = datetime.now(timezone.utc)        
    if args.station != "LOCAL":
        df = fetch_recent_asos(args.station, now)
        write_live_csv(df, csv_path)

    scaler = load_scaler(args.model, args.scaler_npz)
    models = (load_model(args.arch, os.path.join(args.model, "model_1h.pt"), device),
              load_model(args.arch, os.path.join(args.model, "model_24h.pt"), device))

    active_warn = []
    if args.warn_wfo and args.station != "LOCAL":
        active_warn = fetch_active_warnings(args.warn_wfo, args.warn_ugc, now)
    xh, xs, doh, quality = assemble_windows(csv_path)
    
    record, qE = _build_record(args, xh, xs, doh, quality, models, scaler, device,
                               now=now, last_fetch=last_fetch, fetch_start=fetch_start,
                               active_warn=active_warn, backfill=False)

    emb_path = args.embeddings
    append_embedding(emb_path, args.station, record["data_latest_utc"], qE)

    log_path = args.out
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    history = []
    if os.path.exists(log_path):
        try:
            with open(log_path) as f:
                history = json.load(f)
        except Exception:
            history = []
    history.append(record)
    history = rotate_log(history, log_path, args.station, now)   
    tmp = log_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(history, f, indent=2)
    os.replace(tmp, log_path)            
    print(f"[live] {now:%Y-%m-%d %H:%M}Z {args.station} "
          f"coverage={quality['hourly_coverage']} "
          f"-> log now {len(history)} record(s) ({log_path}); embedding -> {emb_path}")
    return record
