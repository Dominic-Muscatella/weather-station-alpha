"""
lstm_experiment.py
==================
Recurrent-bias experiment: keep the conv feature extractor on each leg, but
REPLACE the global avg/max pooling with an LSTM over the conv-feature sequence,
taking the final hidden state (anchored at t, the window's most recent step).

WHY (and the honest prior): global pooling discards *order* — it knows a feature
fired somewhere in the window, not the trajectory. An LSTM keeps the sequence,
so it can in principle read "pressure has been falling for three hours and is
accelerating" rather than just "pressure fell." That's a genuinely different
inductive bias than the conv+pool, worth one clean test.

BUT: the delta experiment already showed that handing the model explicit rate-of
-change didn't move the z frontier. An LSTM is more *capacity* and a different
*temporal bias*, not new *information*. The three surface channels either contain
the convective trigger or they don't, and three experiments (weights, label
widening, deltas) say they mostly don't. So go in expecting this to most likely
CONFIRM the information ceiling — a clean "we tried the recurrent bias too" —
rather than break it. If it does beat B, that's a real and surprising win.

This is LSTM-vs-B: SAME raw 3-channel inputs as B (no deltas), SAME z target,
SAME split/seed/loss/refit. The only change is pooling -> LSTM-final-hidden, on
BOTH legs. So any movement is attributable to the recurrent bias alone.

CAUSALITY: a unidirectional LSTM consumes the window left-to-right; the window
already ends at t and holds only past observations, so the final hidden state
depends only on data <= t. (Bidirectional is deliberately NOT used — it would
muddy the "anchored at t" reading even though all data is <= t.)

Standalone: reuses model._conv_block and train.py helpers; the model keeps the
standard model(x, x2, d) signature on 3-channel normalized input, so scoring
goes through the normal scorecard.fp_fn_scorecard path (no probs shim needed).

Usage:
    from lstm_experiment import run_lstm
    run_lstm("build/dataset.npz", target_1h="z_1h", target_24h="z_24h")
"""
from __future__ import annotations
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

import config as C
from engine.model  import _conv_block
from engine.inference import normalize
from train import chronological_split, get_split, focal_loss, pick_device, _batches

LSTM_HIDDEN = 128       # final-hidden width feeding the per-leg FC (matches the
                        # old avg+max pooled width of 64*2, so the FC/head are
                        # byte-identical to B downstream — only pooling changed)
LSTM_LAYERS = 1
# Pre-LSTM pool length, PER LEG. A raw 576-step BPTT is huge in memory and crawls
# on MPS; pooling the conv features to a shorter sequence keeps the temporal
# trajectory while staying tractable. Values chosen so the input length divides
# EVENLY by the pool length: MPS AdaptiveAvgPool1d only supports divisible sizes
# (168/84 = 2, 576/96 = 6). Non-divisible -> MPS RuntimeError.
LSTM_SEQ_HOURLY = 84    # 168 / 84 = 2
LSTM_SEQ_SUB = 96       # 576 / 96 = 6


class ConvLSTMLeg(nn.Module):
    """Conv stack (identical to ConvLeg's) -> adaptive-pool the feature sequence
    to `seq` steps -> LSTM -> final hidden state -> concat date one-hot ->
    per-leg FC. Swaps ConvLeg's global avg/max pooling for the LSTM; everything
    else matches. The pre-LSTM pool is for tractability (576-step BPTT is huge)."""
    def __init__(self, in_ch=C.N_CHANNELS, widths=C.CONV_CHANNELS, k=C.CONV_KERNEL,
                 date_dim=C.DATE_FEAT_DIM, out_dim=C.LEG_FC,
                 lstm_hidden=LSTM_HIDDEN, lstm_layers=LSTM_LAYERS, seq=LSTM_SEQ_SUB):
        super().__init__()
        chs = [in_ch] + list(widths)
        self.convs = nn.Sequential(*[_conv_block(chs[i], chs[i + 1], k)
                                     for i in range(len(widths))])
        self.pool = nn.AdaptiveAvgPool1d(seq)   # (B, W, L) -> (B, W, seq); seq must
                                                # divide L evenly (MPS constraint)
        self.lstm = nn.LSTM(widths[-1], lstm_hidden, num_layers=lstm_layers,
                            batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden + date_dim, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, series, date_oh):
        x = series.transpose(1, 2)          # (B, L, C) -> (B, C, L) for conv
        x = self.convs(x)                   # (B, W, L)
        x = self.pool(x)                    # (B, W, seq)
        x = x.transpose(1, 2)               # (B, seq, W) for the LSTM
        _, (h, _) = self.lstm(x)            # h: (layers, B, hidden)
        last = h[-1]                        # (B, hidden): state after the final
                                            # (most recent) bin == anchored at t
        x = torch.cat([last, date_oh], dim=1)
        return self.fc(x)


class DualLegLSTM(nn.Module):
    """Same topology as DualLegConvNet, both legs using ConvLSTMLeg. Conv widths,
    LEG_FC, and the head are unchanged — only the time-collapse is now recurrent."""
    def __init__(self, n_outputs=C.N_OUTPUTS, head_fc=C.HEAD_FC, p=C.DROPOUT_P):
        super().__init__()
        self.leg_hourly = ConvLSTMLeg(seq=LSTM_SEQ_HOURLY)   # 7-day hourly, 3-ch
        self.leg_sub = ConvLSTMLeg(seq=LSTM_SEQ_SUB)         # 48-h sub-hourly, 3-ch
        merged = C.LEG_FC * 2
        layers, dims = [], [merged] + list(head_fc)
        for i in range(len(head_fc)):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU(inplace=True), nn.Dropout(p)]
        layers += [nn.Linear(dims[-1], n_outputs)]
        self.head = nn.Sequential(*layers)

    def forward(self, x_hourly, x_sub, date_oh, return_embedding=False):
        a = self.leg_hourly(x_hourly, date_oh)
        b = self.leg_sub(x_sub, date_oh)
        merged = torch.cat([a, b], dim=1)
        # run the head up to (but not including) the final Linear to expose the
        # penultimate embedding the KNN probe reads; final layer maps it to logits
        z = merged
        for layer in self.head[:-1]:
            z = layer(z)
        logits = self.head[-1](z)
        if return_embedding:
            return logits, z          # z = penultimate activation (pre-final-Linear)
        return logits


def train_lstm(data: dict, target: str, twenty_four: bool, device=None,
               epochs=C.EPOCHS, verbose=True, refit: Optional[bool] = None):
    """Train the LSTM model on `target` (e.g. 'z_1h'/'z_24h'). Mirrors train_one
    exactly (raw 3-channel normalize, split, focal loss, refit) so it's directly
    comparable to B. Returns (model, metrics)."""
    refit = C.REFIT_ON_TRAINVAL if refit is None else refit
    torch.manual_seed(C.SEED); np.random.seed(C.SEED)
    rng = np.random.default_rng(C.SEED)
    device = device if isinstance(device, torch.device) else pick_device(device)

    mean, std = data["scaler_mean"], data["scaler_std"]
    X = normalize(data["X"], mean, std).astype(np.float32)
    X2 = normalize(data["X2"], mean, std).astype(np.float32)
    DOH = data["date_oh"].astype(np.float32)
    Y = data[target].astype(np.float32)
    W = (data["weight"] if twenty_four else data["weight_hour"]).astype(np.float32)

    sp = get_split(data)
    if verbose:
        print(f"[lstm:{target}] device={device.type}  hidden={LSTM_HIDDEN} "
              f"layers={LSTM_LAYERS} seq=(h{LSTM_SEQ_HOURLY},s{LSTM_SEQ_SUB})  "
              f"train={len(sp.tr)} val={len(sp.va)} test={len(sp.te)}")

    tX = torch.tensor(X, device=device); tX2 = torch.tensor(X2, device=device)
    tD = torch.tensor(DOH, device=device)
    tY = torch.tensor(Y, device=device); tW = torch.tensor(W, device=device)

    def make():
        m = DualLegLSTM().to(device)
        o = torch.optim.Adam(m.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
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
    last_lr = opt.param_groups[0]["lr"]
    for ep in range(epochs):
        tr_loss = epoch_pass(model, opt, sp.tr, True)
        if have_val:
            va_loss = epoch_pass(model, opt, sp.va, False)
            sched.step(va_loss)
            lr_now = opt.param_groups[0]["lr"]
            lr_dropped = lr_now < last_lr            # scheduler just cut the LR
            last_lr = lr_now
            if verbose:
                print(f"[lstm:{target}] epoch {ep:02d}  train={tr_loss:.4f}  "
                      f"val={va_loss:.4f}  lr={lr_now:.1e}"
                      + ("   * reloaded best" if lr_dropped and best_state is not None else ""))
            if va_loss < best_val - 1e-6:
                best_val, best_ep, patience = va_loss, ep, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience += 1
                # On an LR drop, restart from the best weights so the smaller LR
                # refines the best point reached so far rather than continuing from
                # a possibly-worse current point (matches train_one's behavior).
                if lr_dropped and best_state is not None:
                    model.load_state_dict({k: v.to(next(model.parameters()).device)
                                           for k, v in best_state.items()})
                if patience >= C.EARLY_STOP_PATIENCE:
                    if verbose:
                        print(f"[lstm:{target}] early stop {ep} (best {best_ep})")
                    break
        else:
            best_ep = ep
    if best_state is not None:
        model.load_state_dict({k: v.to(next(model.parameters()).device)
                               for k, v in best_state.items()})

    if refit and have_val:
        n_ref = best_ep + 1
        trval = np.concatenate([sp.tr, sp.va])
        if verbose:
            print(f"[lstm:{target}] refit on train+val ({len(trval)}) for {n_ref} epochs")
        for ep in range(n_ref):
            tr_loss = epoch_pass(model, opt, trval, True)
            sched.step(tr_loss)
            if verbose:
                print(f"[lstm:{target}] refit epoch {ep:02d}  train={tr_loss:.4f}")

    metrics = {"target": target, "best_epoch": int(best_ep), "best_val_loss": best_val,
               "lstm_hidden": LSTM_HIDDEN, "lstm_layers": LSTM_LAYERS}
    return model, metrics


def run_lstm(npz_path: str, target_1h="z_1h", target_24h="z_24h",
             device=None, epochs=C.EPOCHS, refit=None, save_dir=None):
    """Train LSTM models on the targets and score on z via the normal scorecard.
    If save_dir is given, persist each model's state_dict there (model_1h.pt /
    model_24h.pt) so the KNN probe and the live GUI can reload the encoder."""
    from scorecard import fp_fn_scorecard
    dev = device if isinstance(device, torch.device) else pick_device(device)
    print(f"[lstm] device={dev.type}  targets=({target_1h},{target_24h})  epochs={epochs}")
    data = dict(np.load(npz_path, allow_pickle=True))
    m1, met1 = train_lstm(data, target_1h, twenty_four=False, device=dev, epochs=epochs, refit=refit)
    m24, met24 = train_lstm(data, target_24h, twenty_four=True, device=dev, epochs=epochs, refit=refit)
    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
        torch.save(m1.state_dict(), os.path.join(save_dir, "model_1h.pt"))
        torch.save(m24.state_dict(), os.path.join(save_dir, "model_24h.pt"))
        print(f"[lstm] saved encoders -> {save_dir}/model_{{1h,24h}}.pt")
    print(f"\n>>> LSTM model trained on {target_1h}, scored on z_1h:")
    fp_fn_scorecard(m1, data, target_1h)
    print(f"\n>>> LSTM model trained on {target_24h}, scored on z_24h:")
    fp_fn_scorecard(m24, data, target_24h)
    return {"model_1h": m1, "model_24h": m24, "metrics": {target_1h: met1, target_24h: met24}}


if __name__ == "__main__":
    import os
    run_lstm(os.path.join(C.BUILD_DIR, "dataset.npz"), target_1h="z_1h", target_24h="z_24h",
             save_dir="model_package_lstm")
