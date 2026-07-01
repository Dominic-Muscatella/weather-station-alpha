from __future__ import annotations
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

import config as C
from engine.model  import _conv_block
from train import get_split, focal_loss, pick_device, _batches
from engine.models.delta_experiment import (add_deltas, fit_channel_scaler, _scale,
                              HOURLY_IN, SUB_IN, SUB_KERNEL)
from engine.models.lstm_experiment import LSTM_HIDDEN, LSTM_LAYERS, LSTM_SEQ_HOURLY, LSTM_SEQ_SUB


class ConvLSTMLegDelta(nn.Module):

    def __init__(self, in_ch, k=C.CONV_KERNEL, widths=C.CONV_CHANNELS,
                 date_dim=C.DATE_FEAT_DIM, out_dim=C.LEG_FC,
                 lstm_hidden=LSTM_HIDDEN, lstm_layers=LSTM_LAYERS, seq=LSTM_SEQ_SUB):
        super().__init__()
        chs = [in_ch] + list(widths)
        self.convs = nn.Sequential(*[_conv_block(chs[i], chs[i + 1], k)
                                     for i in range(len(widths))])
        self.pool = nn.AdaptiveAvgPool1d(seq)
        self.lstm = nn.LSTM(widths[-1], lstm_hidden, num_layers=lstm_layers, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(lstm_hidden + date_dim, out_dim), nn.ReLU(inplace=True))

    def forward(self, series, date_oh):
        x = series.transpose(1, 2)
        x = self.convs(x)
        x = self.pool(x)
        x = x.transpose(1, 2)
        _, (h, _) = self.lstm(x)
        last = h[-1]
        return self.fc(torch.cat([last, date_oh], dim=1))


class DualLegLSTMDelta(nn.Module):

    def __init__(self, hourly_in=HOURLY_IN, sub_in=SUB_IN, sub_kernel=SUB_KERNEL,
                 n_outputs=C.N_OUTPUTS, head_fc=C.HEAD_FC, p=C.DROPOUT_P):
        super().__init__()
        self.leg_hourly = ConvLSTMLegDelta(in_ch=hourly_in, k=C.CONV_KERNEL, seq=LSTM_SEQ_HOURLY)
        self.leg_sub = ConvLSTMLegDelta(in_ch=sub_in, k=sub_kernel, seq=LSTM_SEQ_SUB)
        merged = C.LEG_FC * 2
        layers, dims = [], [merged] + list(head_fc)
        for i in range(len(head_fc)):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU(inplace=True), nn.Dropout(p)]
        layers += [nn.Linear(dims[-1], n_outputs)]
        self.head = nn.Sequential(*layers)

    def forward(self, x_hourly, x_sub, date_oh, return_embedding=False):
        a = self.leg_hourly(x_hourly, date_oh)
        b = self.leg_sub(x_sub, date_oh)
        z = torch.cat([a, b], dim=1)
        for layer in self.head[:-1]:
            z = layer(z)
        logits = self.head[-1](z)
        if return_embedding:
            return logits, z
        return logits


def train_lstm_delta(data, target, twenty_four, device=None, epochs=C.EPOCHS,
                     verbose=True, refit: Optional[bool] = None):
    refit = C.REFIT_ON_TRAINVAL if refit is None else refit
    torch.manual_seed(C.SEED); np.random.seed(C.SEED)
    rng = np.random.default_rng(C.SEED)
    device = device if isinstance(device, torch.device) else pick_device(device)

    Xd, X2d = add_deltas(data["X"], data["X2"])
    DOH = data["date_oh"].astype(np.float32)
    Y = data[target].astype(np.float32)
    W = (data["weight"] if twenty_four else data["weight_hour"]).astype(np.float32)

    sp = get_split(data)
    mu_h, sd_h = fit_channel_scaler(Xd, sp.tr)         
    mu_s, sd_s = fit_channel_scaler(X2d, sp.tr)
    Xd = _scale(Xd, mu_h, sd_h); X2d = _scale(X2d, mu_s, sd_s)
    if verbose:
        print(f"[lstm+delta:{target}] device={device.type}  hourly_in={HOURLY_IN} "
              f"sub_in={SUB_IN} sub_k={SUB_KERNEL}  seq=(h{LSTM_SEQ_HOURLY},s{LSTM_SEQ_SUB})  "
              f"train={len(sp.tr)} val={len(sp.va)} test={len(sp.te)}")
        print(f"[lstm+delta:{target}] hourly pressure-delta scaler sd (should be ~O(0.1-3), not crushed): "
              f"{np.round(sd_h[4:9],3)}")

    tX = torch.tensor(Xd, device=device); tX2 = torch.tensor(X2d, device=device)
    tD = torch.tensor(DOH, device=device)
    tY = torch.tensor(Y, device=device); tW = torch.tensor(W, device=device)

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

    model = DualLegLSTMDelta().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
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
            lr_now = opt.param_groups[0]["lr"]; lr_dropped = lr_now < last_lr; last_lr = lr_now
            if verbose:
                print(f"[lstm+delta:{target}] epoch {ep:02d}  train={tr_loss:.4f}  "
                      f"val={va_loss:.4f}  lr={lr_now:.1e}"
                      + ("   * reloaded best" if lr_dropped and best_state is not None else ""))
            if va_loss < best_val - 1e-6:
                best_val, best_ep, patience = va_loss, ep, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience += 1
                if lr_dropped and best_state is not None:
                    model.load_state_dict({k: v.to(next(model.parameters()).device)
                                           for k, v in best_state.items()})
                if patience >= C.EARLY_STOP_PATIENCE:
                    if verbose:
                        print(f"[lstm+delta:{target}] early stop {ep} (best {best_ep})")
                    break
        else:
            best_ep = ep
    if best_state is not None:
        model.load_state_dict({k: v.to(next(model.parameters()).device) for k, v in best_state.items()})

    if refit and have_val:
        n_ref = best_ep + 1
        trval = np.concatenate([sp.tr, sp.va])
        if verbose:
            print(f"[lstm+delta:{target}] refit on train+val ({len(trval)}) for {n_ref} epochs")
        for ep in range(n_ref):
            tr_loss = epoch_pass(model, opt, trval, True)
            sched.step(tr_loss)
            if verbose:
                print(f"[lstm+delta:{target}] refit epoch {ep:02d}  train={tr_loss:.4f}")

    scalers = {"hourly_mean": mu_h, "hourly_std": sd_h, "sub_mean": mu_s, "sub_std": sd_s}
    metrics = {"target": target, "best_epoch": int(best_ep), "best_val_loss": best_val}
    return model, metrics, scalers


def run_lstm_delta(npz_path, target_1h="z_1h", target_24h="z_24h", device=None,
                   epochs=C.EPOCHS, refit=None, save_dir=None):
    from scorecard import _scorecard_from_probs
    import os
    dev = device if isinstance(device, torch.device) else pick_device(device)
    print(f"[lstm+delta] device={dev.type}  targets=({target_1h},{target_24h})  epochs={epochs}")
    data = dict(np.load(npz_path, allow_pickle=True))
    sp = get_split(data)
    labels = [str(x) for x in data["labels"]]
    out = {}
    for target, t24 in [(target_1h, False), (target_24h, True)]:
        model, met, sc = train_lstm_delta(data, target, twenty_four=t24, device=dev,
                                          epochs=epochs, refit=refit)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(),
                       os.path.join(save_dir, f"model_{'24h' if t24 else '1h'}.pt"))
        
        Xd, X2d = add_deltas(data["X"], data["X2"])
        Xd = _scale(Xd, sc["hourly_mean"], sc["hourly_std"])
        X2d = _scale(X2d, sc["sub_mean"], sc["sub_std"])
        model.eval()
        with torch.no_grad():
            te = sp.te
            xb = torch.tensor(Xd[te], device=dev); x2b = torch.tensor(X2d[te], device=dev)
            db = torch.tensor(data["date_oh"][te].astype(np.float32), device=dev)
            probs = torch.sigmoid(model(xb, x2b, db)).cpu().numpy()
        zk = "z_1h" if target.endswith("1h") else "z_24h"
        print(f"\n>>> LSTM+delta trained on {target}, scored on {zk}:")
        _scorecard_from_probs(probs, data[zk][te], labels, title=f"lstm+delta {target}")
        out[target] = (model, met, sc)
    if save_dir:
        print(f"[lstm+delta] saved encoders -> {save_dir}")
    return out


if __name__ == "__main__":
    import os
    run_lstm_delta(os.path.join(C.BUILD_DIR, "dataset_multi.npz"),
                   target_1h="z_1h", target_24h="z_24h", save_dir="model_package_lstm_delta")
