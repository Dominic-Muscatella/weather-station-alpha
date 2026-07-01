from download_data import _get, IEM_ASOS
import io
import os
import config as C
from data_pipeline import load_warnings
from datetime import datetime, timedelta
import pandas as pd


def fetch_asos_range(station: str, start_utc: datetime, end_utc: datetime) -> pd.DataFrame:

    url = C.IEM_ASOS_RANGE.format(
        st=station,
        y1=start_utc.year, m1=start_utc.month, d1=start_utc.day, h1=start_utc.hour,
        y2=end_utc.year, m2=end_utc.month, d2=end_utc.day, h2=end_utc.hour,
    )
    raw = _get(url, timeout=300)
    d = pd.read_csv(io.BytesIO(raw), na_values=["M", "T", ""])
    if not len(d):
        raise RuntimeError(f"IEM returned no rows for {station} "
                           f"({start_utc:%Y-%m-%d}..{end_utc:%Y-%m-%d})")
    mslp = pd.to_numeric(d.get("mslp"), errors="coerce")
    alti = pd.to_numeric(d.get("alti"), errors="coerce") * 33.8638866667
    d["pressure_hpa"] = mslp.where(mslp.notna(), alti)
    return d[["valid", "tmpf", "relh", "pressure_hpa"]]


def fetch_recent_asos(station: str, now_utc: datetime) -> pd.DataFrame:
    
    return fetch_asos_range(station, now_utc - timedelta(days=C.FETCH_DAYS), now_utc)

def write_live_csv(df: pd.DataFrame, path: str):
    
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False)
    return path


def fetch_warning_intervals(wfo, ugc_keep, start_utc, end_utc):
    sts = start_utc.strftime("%Y-%m-%dT%H:%MZ")
    ets = (end_utc + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%MZ")
    url = C.IEM_WARN_RANGE.format(sts=sts, ets=ets, wfo=wfo)
    try:
        raw = _get(url, timeout=300)
        tmp = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"_bf_warn_{wfo}.csv")
        with open(tmp, "wb") as f:
            f.write(raw)
        wdf = load_warnings(tmp, ugc_keep=tuple(ugc_keep) if ugc_keep else ())
        if wdf is None or not len(wdf):
            return []
        labels = list(C.ALL_LABELS)
        out = []
        for _, r in wdf.iterrows():
            out.append({"start": r["start"], "end": r["end"], "phen": str(r["phen"]),
                        "classes": [labels[i] for i in (r["cls"] or ())]})
        return out
    except Exception as e:
        print(f"[backfill] warning span fetch failed ({e!r}); no ground truth in replay")
        return []


def warnings_active_at(intervals, when_dt):
    when = pd.Timestamp(when_dt)
    if when.tzinfo is not None:
        when = when.tz_convert("UTC").tz_localize(None)
    out = []
    for iv in intervals:
        s, e = pd.Timestamp(iv["start"]), pd.Timestamp(iv["end"])
        s = s.tz_convert("UTC").tz_localize(None) if s.tzinfo is not None else s
        e = e.tz_convert("UTC").tz_localize(None) if e.tzinfo is not None else e
        if s <= when <= e:
            out.append({"phen": iv["phen"], "classes": iv["classes"],
                        "start": s.isoformat(), "end": e.isoformat()})
    return out


def fetch_active_warnings(wfo, ugc_keep, now_utc, lookback_days=2):
    
    sts = (now_utc - timedelta(days=lookback_days)).strftime("%Y-%m-%dT%H:%MZ")
    ets = (now_utc + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%MZ")
    url = C.IEM_WARN_RANGE.format(sts=sts, ets=ets, wfo=wfo)
    try:
        raw = _get(url, timeout=120)
        tmp = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"_live_warn_{wfo}.csv")
        with open(tmp, "wb") as f:
            f.write(raw)
        wdf = load_warnings(tmp, ugc_keep=tuple(ugc_keep) if ugc_keep else ())
        if wdf is None or not len(wdf):
            return []
        now_ts = pd.Timestamp(now_utc)
        active = wdf[(wdf["start"] <= now_ts) & (wdf["end"] >= now_ts)]
        labels = list(C.ALL_LABELS)
        out = []
        for _, r in active.iterrows():
            classes = [labels[i] for i in (r["cls"] or ())]
            out.append({"phen": str(r["phen"]), "classes": classes,
                        "start": r["start"].isoformat(), "end": r["end"].isoformat()})
        return out
    except Exception as e:
        print(f"[live] warning fetch failed ({e!r}); overlay has no ground truth this cycle")
        return []


