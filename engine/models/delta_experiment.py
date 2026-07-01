from __future__ import annotations
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

import config as C
from engine.model  import ConvLeg
from train import chronological_split, focal_loss, pick_device, _batches



_TEMP, _PRESS, _HUM = 0, 1, 2


HOURLY_DELTAS = [
    (_TEMP,  1, "temp_d1h"), (_TEMP,  6, "temp_d6h"), (_TEMP, 24, "temp_d24h"),
    (_PRESS, 1, "press_d1h"), (_PRESS, 6, "press_d6h"), (_PRESS, 12, "press_d12h"),
    (_PRESS, 24, "press_d24h"),
    (_HUM,   1, "hum_d1h"), (_HUM,   6, "hum_d6h"), (_HUM,  24, "hum_d24h"),
]

SUB_DELTAS = [
    (_TEMP, 12, "temp_d1h"),
    (_PRESS, 1, "press_d5m"), (_PRESS, 3, "press_d15m"), (_PRESS, 6, "press_d30m"),
    (_PRESS, 12, "press_d1h"), (_PRESS, 72, "press_d6h"),
    (_HUM,  12, "hum_d1h"),
]

HOURLY_ORDER = ["temp_raw", "temp_d1h", "temp_d6h", "temp_d24h",
                "press_raw", "press_d1h", "press_d6h", "press_d12h", "press_d24h",
                "hum_raw", "hum_d1h", "hum_d6h", "hum_d24h"]
SUB_ORDER = ["temp_raw", "temp_d1h",
             "press_raw", "press_d5m", "press_d15m", "press_d30m", "press_d1h", "press_d6h",
             "hum_raw", "hum_d1h"]
HOURLY_IN = len(HOURLY_ORDER)   
SUB_IN = len(SUB_ORDER)         
SUB_KERNEL = 9                  


def _backward_delta(W: np.ndarray, ch: int, step: int) -> np.ndarray:
    """W is (N, L, C_raw). Returns (N, L) = W[:,t,ch] - W[:,t-step,ch], a strictly
    backward difference along the time axis. The first `step` positions can't look
    back, so they're EDGE-padded with the delta at position `step` (no future
    value is ever used)."""
    v = W[:, :, ch]                              
    d = np.empty_like(v)
    d[:, step:] = v[:, step:] - v[:, :-step]     
    d[:, :step] = d[:, step:step + 1]            
    return d


def _assemble(W: np.ndarray, raw_idx, deltas, order) -> np.ndarray:
    """Build the delta-augmented (N, L, C_out) array in `order`. raw_idx maps a
    raw name -> its column in W; deltas is the spec list."""
    N, L, _ = W.shape
    cols = {}
    for name, ci in raw_idx.items():
        cols[name] = W[:, :, ci]
    for ci, step, name in deltas:
        cols[name] = _backward_delta(W, ci, step)
    return np.stack([cols[k] for k in order], axis=2).astype(np.float32)


def add_deltas(X: np.ndarray, X2: np.ndarray):
    
    raw = {"temp_raw": _TEMP, "press_raw": _PRESS, "hum_raw": _HUM}
    Xd = _assemble(X, raw, HOURLY_DELTAS, HOURLY_ORDER)
    X2d = _assemble(X2, raw, SUB_DELTAS, SUB_ORDER)
    return Xd, X2d


def assert_causal():

    rng = np.random.default_rng(0)
    X = rng.standard_normal((4, C.HOURLY_WINDOW_LEN, 3)).astype(np.float32)
    X2 = rng.standard_normal((4, C.SUBHOURLY_WINDOW_LEN, 3)).astype(np.float32)
    Xd0, X2d0 = add_deltas(X, X2)
    Xp = X.copy(); Xp[:, -1, :] += 1000.0        
    X2p = X2.copy(); X2p[:, -1, :] += 1000.0
    Xd1, X2d1 = add_deltas(Xp, X2p)
    
    if not np.allclose(Xd0[:, :-1, :], Xd1[:, :-1, :]):
        raise AssertionError("HOURLY delta leak: a past position changed when t+ was perturbed")
    if not np.allclose(X2d0[:, :-1, :], X2d1[:, :-1, :]):
        raise AssertionError("SUBHOURLY delta leak: a past position changed when t+ was perturbed")
    return True


def fit_channel_scaler(arr: np.ndarray, train_idx: np.ndarray):

    tr = arr[train_idx]                           
    mu = tr.mean(axis=(0, 1), dtype=np.float64)
    sd = tr.std(axis=(0, 1), dtype=np.float64)
    sd[sd == 0] = 1.0
    return mu.astype(np.float32), sd.astype(np.float32)


class DualLegDelta(nn.Module):

    def __init__(self, hourly_in=HOURLY_IN, sub_in=SUB_IN, sub_kernel=SUB_KERNEL,
                 n_outputs=C.N_OUTPUTS, head_fc=C.HEAD_FC, p=C.DROPOUT_P):
        super().__init__()
        self.leg_hourly = ConvLeg(in_ch=hourly_in, k=C.CONV_KERNEL)
        self.leg_sub = ConvLeg(in_ch=sub_in, k=sub_kernel)
        merged = C.LEG_FC * 2
        layers, dims = [], [merged] + list(head_fc)
        for i in range(len(head_fc)):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU(inplace=True), nn.Dropout(p)]
        layers += [nn.Linear(dims[-1], n_outputs)]
        self.head = nn.Sequential(*layers)

    def forward(self, x_hourly, x_sub, date_oh):
        a = self.leg_hourly(x_hourly, date_oh)
        b = self.leg_sub(x_sub, date_oh)
        return self.head(torch.cat([a, b], dim=1))


def _scale(arr, mu, sd):
    return (arr - mu) / sd


def train_delta(data: dict, target: str, twenty_four: bool, device=None,
                epochs=C.EPOCHS, verbose=True, refit: Optional[bool] = None,
                init_state_path: Optional[str] = None, lr_scale: float = 1.0):

    refit = C.REFIT_ON_TRAINVAL if refit is None else refit
    torch.manual_seed(C.SEED); np.random.seed(C.SEED)
    rng = np.random.default_rng(C.SEED)
    device = device if isinstance(device, torch.device) else pick_device(device)

    Xd, X2d = add_deltas(data["X"], data["X2"])
    DOH = data["date_oh"].astype(np.float32)
    Y = data[target].astype(np.float32)
    W = (data["weight"] if twenty_four else data["weight_hour"]).astype(np.float32)

    sp = chronological_split(data["t_sec"])
    mu_h, sd_h = fit_channel_scaler(Xd, sp.tr)
    mu_s, sd_s = fit_channel_scaler(X2d, sp.tr)
    Xd = _scale(Xd, mu_h, sd_h); X2d = _scale(X2d, mu_s, sd_s)
    if verbose:
        lr = C.LR * lr_scale
        print(f"[delta:{target}] device={device.type}  hourly_in={HOURLY_IN} "
              f"sub_in={SUB_IN} sub_k={SUB_KERNEL}  lr={lr:.2e}  "
              f"train={len(sp.tr)} val={len(sp.va)} test={len(sp.te)}"
              + (f"  init={os.path.basename(init_state_path)}" if init_state_path else ""))

    tX = torch.tensor(Xd, device=device); tX2 = torch.tensor(X2d, device=device)
    tD = torch.tensor(DOH, device=device)
    tY = torch.tensor(Y, device=device); tW = torch.tensor(W, device=device)

    def make():
        m = DualLegDelta().to(device)
        if init_state_path:
            m.load_state_dict(torch.load(init_state_path, map_location=device))
        o = torch.optim.Adam(m.parameters(), lr=C.LR * lr_scale, weight_decay=C.WEIGHT_DECAY)
        return m, o

    def epoch_pass(model, opt, idx, train):
        model.train(train)
        total, count = 0.0, 0
        for bi in _batches(len(idx), C.BATCH_SIZE, shuffle=train, rng=rng):
            sel = torch.as_tensor(idx[bi], dtype=torch.long, device=device)
            xb = tX[sel]; x2b = tX2[sel]; db = tD[sel]; yb = tY[sel]; wb = tW[sel]
            if train:
                opt.zero_grad()
            with torch.set_grad_enabled(train):
                logits = model(xb, x2b, db)
                loss = (focal_loss(logits, yb).mean(dim=1) * wb).sum() / wb.sum()
                if train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP_NORM)
                    opt.step()
            total += loss.item() * len(bi); count += len(bi)
        return total / max(count, 1)

    model, opt = make()
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, factor=C.LR_PLATEAU_FACTOR, patience=C.LR_PLATEAU_PATIENCE)
    best_val, best_state, best_ep, patience = float("inf"), None, 0, 0
    have_val = len(sp.va) > 0
    for ep in range(epochs):
        tr_loss = epoch_pass(model, opt, sp.tr, True)
        if have_val:
            va_loss = epoch_pass(model, opt, sp.va, False)
            sched.step(va_loss)
            if verbose:
                print(f"[delta:{target}] epoch {ep:02d}  train={tr_loss:.4f}  "
                      f"val={va_loss:.4f}  lr={opt.param_groups[0]['lr']:.1e}")
            if va_loss < best_val - 1e-6:
                best_val, best_ep, patience = va_loss, ep, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience += 1
                if patience >= C.EARLY_STOP_PATIENCE:
                    if verbose:
                        print(f"[delta:{target}] early stop {ep} (best {best_ep})")
                    break
        else:
            best_ep = ep
    if best_state is not None:
        model.load_state_dict(best_state)

    if refit and have_val:
        n_ref = best_ep + 1
        trval = np.concatenate([sp.tr, sp.va])
        if verbose:
            print(f"[delta:{target}] refit on train+val ({len(trval)}) for {n_ref} epochs")
        for ep in range(n_ref):
            tr_loss = epoch_pass(model, opt, trval, True)
            sched.step(tr_loss)
            if verbose:
                print(f"[delta:{target}] refit epoch {ep:02d}  train={tr_loss:.4f}")

    scalers = {"hourly_mean": mu_h, "hourly_std": sd_h, "sub_mean": mu_s, "sub_std": sd_s}
    metrics = {"target": target, "best_epoch": int(best_ep), "best_val_loss": best_val,
               "hourly_in": HOURLY_IN, "sub_in": SUB_IN}
    return model, metrics, scalers



class _DeltaScored:
    """Adapts a trained delta model so scorecard.fp_fn_scorecard can call it like
    a normal model. scorecard recomputes probs from data["X"]/["X2"]; we intercept
    by precomputing the scaled delta tensors and overriding the forward path is
    not possible — so we instead expose a model whose forward takes the SAME raw
    inputs scorecard will pass, applies deltas+scaling inside. See run_delta."""
    pass


def run_delta(npz_path: str, target_1h="y_1h", target_24h="y_24h",
              device=None, epochs=C.EPOCHS, refit=None,
              init_dir: Optional[str] = None, lr_scale: float = 1.0):

    from scorecard import _scorecard_from_probs   
    dev = device if isinstance(device, torch.device) else pick_device(device)
    print(f"[delta] device={dev.type}  targets=({target_1h},{target_24h})  "
          f"epochs={epochs}  lr_scale={lr_scale}"
          + (f"  init_dir={init_dir}" if init_dir else ""))
    assert_causal()
    print("[delta] causality check passed (no future leak in backward diffs)")
    data = dict(np.load(npz_path, allow_pickle=True))
    sp = chronological_split(data["t_sec"])

    out = {}
    for target, t24, zk in [(target_1h, False, "z_1h"), (target_24h, True, "z_24h")]:
        init = os.path.join(init_dir, f"model_{'24h' if t24 else '1h'}.pt") if init_dir else None
        model, met, sc = train_delta(data, target, twenty_four=t24, device=dev,
                                     epochs=epochs, refit=refit,
                                     init_state_path=init, lr_scale=lr_scale)
        
        Xd, X2d = add_deltas(data["X"], data["X2"])
        Xd = _scale(Xd, sc["hourly_mean"], sc["hourly_std"])
        X2d = _scale(X2d, sc["sub_mean"], sc["sub_std"])
        model.eval()
        with torch.no_grad():
            te = sp.te
            xb = torch.tensor(Xd[te], device=dev)
            x2b = torch.tensor(X2d[te], device=dev)
            db = torch.tensor(data["date_oh"][te].astype(np.float32), device=dev)
            probs = torch.sigmoid(model(xb, x2b, db)).cpu().numpy()
        Z = data[zk][te]
        print(f"\n>>> delta model trained on {target}, scored on {zk}:")
        _scorecard_from_probs(probs, Z, list(data["labels"]))
        out[target] = (model, met, sc)
    return out


if __name__ == "__main__":
    npz = os.path.join(C.BUILD_DIR, "dataset.npz")
    
    
    run_delta(npz, target_1h="z_1h", target_24h="z_24h")
