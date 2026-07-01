import os
import json
import numpy as np
import pandas as pd
import torch
from engine.inference import normalize
from engine.model import _disable_mc_dropout


def append_embedding(emb_path, station, data_latest_utc, embedding):
    os.makedirs(os.path.dirname(emb_path) or ".", exist_ok=True)
    rec = {"station": station, "data_latest_utc": data_latest_utc,
           "embedding": [float(v) for v in embedding]}
    with open(emb_path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def lookup_embedding(emb_path, station, data_latest_utc):
    if not os.path.exists(emb_path):
        return None
    found = None
    with open(emb_path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("station") == station and r.get("data_latest_utc") == data_latest_utc:
                found = r.get("embedding")
    return found


@torch.no_grad()
def embed(model, xh, xs, doh, scaler_mean, scaler_std, device):
    print(f"[model run] normalizing Xh...")
    Xh = normalize(xh[None], scaler_mean, scaler_std).astype(np.float32)
    print(f"[model run] normalizing Xs...")
    Xs = normalize(xs[None], scaler_mean, scaler_std).astype(np.float32)
    print(f"[model run] eval mode...")
    model.eval()
    print(f"[model run] th to Tensor...")
    th = torch.tensor(Xh, device=device); ts = torch.tensor(Xs, device=device)
    print(f"[model run] th to Tensor...")
    td = torch.tensor(doh[None], device=device)
    print(f"[model run] disable dropout...")
    _disable_mc_dropout(model)
    print(f"[model run] embedding...")
    emb = model(th, ts, td, return_embedding=True)
    print(f"[model run] embedded!")
    return emb.cpu().numpy()[0]


def load_knn_refs(csv_path, n_out):
    df = pd.read_csv(csv_path)
    emb_cols = sorted([c for c in df.columns if c.startswith("emb_")],
                      key=lambda s: int(s.split("_")[1]))
    ref_E = df[emb_cols].to_numpy(dtype=np.float64)
    ref_cls = df["class"].to_numpy(dtype=int)
    if any(c.startswith("prior_") for c in df.columns):
        priors = np.array([df[f"prior_{c}"].iloc[0] for c in range(n_out)], dtype=np.float64)
    else:
        counts = np.bincount(ref_cls, minlength=n_out).astype(np.float64)
        priors = counts / max(counts.sum(), 1)
    return ref_E, ref_cls, priors


def knn_live(query_E, ref_E, ref_cls, n_out, k, prior=None, eps=1e-6):
    d = np.linalg.norm(ref_E - query_E[None, :], axis=1)
    kk = min(k, len(ref_E))
    nn = np.argpartition(d, kk - 1)[:kk]
    dd = d[nn]; w = 1.0 / (dd + eps); cls = ref_cls[nn]
    scores = np.zeros(n_out)
    for c in range(n_out):
        scores[c] = w[cls == c].sum()
    conf = float(1.0 / (1.0 + dd.mean()))
    if prior is not None:
        scores = scores * prior
    s = scores.sum()
    scores = scores / s if s > 0 else scores
    return scores, conf