from __future__ import annotations
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import copy
import config as C
from engine.model  import _conv_block
from engine.inference import normalize
from train import get_split, focal_loss, pick_device, _batches
from engine.models.lstm_experiment import LSTM_HIDDEN, LSTM_LAYERS, LSTM_SEQ_HOURLY, LSTM_SEQ_SUB

DATE_FUZZ_PROB = 0.30        
                             


def fuzz_month_onehot(doh_batch: np.ndarray, rng, prob=DATE_FUZZ_PROB):

    out = doh_batch.copy()
    n = len(out)
    pick = rng.random(n) < prob
    if not pick.any():
        return out
    shifts = rng.choice((-1, 1), size=pick.sum())
    rows = np.flatnonzero(pick)
    for r, s in zip(rows, shifts):
        out[r] = np.roll(out[r], s)        
    return out


class ConvAttnLSTMLeg(nn.Module):

    def __init__(self, in_ch=C.N_CHANNELS, widths=C.CONV_CHANNELS, k=C.CONV_KERNEL,
                 date_dim=C.DATE_FEAT_DIM, out_dim=C.LEG_FC, lstm_hidden=LSTM_HIDDEN,
                 lstm_layers=LSTM_LAYERS, seq=LSTM_SEQ_SUB, use_attention=False):
        super().__init__()
        chs = [in_ch] + list(widths)
        self.convs = nn.Sequential(*[_conv_block(chs[i], chs[i + 1], k)
                                     for i in range(len(widths))])
        self.pool = nn.AdaptiveAvgPool1d(seq)
        self.lstm = nn.LSTM(widths[-1], lstm_hidden, num_layers=lstm_layers, batch_first=True)
        self.use_attention = use_attention
        if use_attention:
            
            self.attn = nn.Linear(lstm_hidden, 1)
        self.fc = nn.Sequential(nn.Linear(lstm_hidden + date_dim, out_dim), nn.ReLU(inplace=True))

    def forward(self, series, date_oh):
        x = series.transpose(1, 2)
        x = self.convs(x)
        x = self.pool(x)
        x = x.transpose(1, 2)               
        outs, (h, _) = self.lstm(x)         
        if self.use_attention:
            scores = self.attn(outs).squeeze(-1)        
            wts = torch.softmax(scores, dim=1).unsqueeze(-1)  
            ctx = (outs * wts).sum(dim=1)               
        else:
            ctx = h[-1]                                 
        return self.fc(torch.cat([ctx, date_oh], dim=1))


class DualLegAttnLSTM(nn.Module):
    def __init__(self, n_outputs=C.N_OUTPUTS, head_fc=C.HEAD_FC, p=C.DROPOUT_P,
                 use_attention=False):
        super().__init__()
        self.leg_hourly = ConvAttnLSTMLeg(seq=LSTM_SEQ_HOURLY, use_attention=use_attention)
        self.leg_sub = ConvAttnLSTMLeg(seq=LSTM_SEQ_SUB, use_attention=use_attention)
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
        print("[model run] ver 1")
        if return_embedding:
            print("[model run] detaching embeddings...")
            return z
        logits = self.head[-1](z)
        return logits


def train_lstm_attn(data, target, twenty_four, device=None, epochs=C.EPOCHS,
                    verbose=True, refit: Optional[bool] = None,
                    use_date_fuzz=False, use_attention=False):
    refit = C.REFIT_ON_TRAINVAL if refit is None else refit
    torch.manual_seed(C.SEED); np.random.seed(C.SEED)
    rng = np.random.default_rng(C.SEED)
    fuzz_rng = np.random.default_rng(C.SEED + 1)
    device = device if isinstance(device, torch.device) else pick_device(device)

    mean, std = data["scaler_mean"], data["scaler_std"]
    X = normalize(data["X"], mean, std).astype(np.float32)
    X2 = normalize(data["X2"], mean, std).astype(np.float32)
    DOH = data["date_oh"].astype(np.float32)
    Y = data[target].astype(np.float32)
    W = (data["weight"] if twenty_four else data["weight_hour"]).astype(np.float32)

    sp = get_split(data)
    if verbose:
        print(f"[lstm-attn:{target}] device={device.type}  attention={use_attention} "
              f"date_fuzz={use_date_fuzz}  seq=(h{LSTM_SEQ_HOURLY},s{LSTM_SEQ_SUB})  "
              f"train={len(sp.tr)} val={len(sp.va)} test={len(sp.te)}")

    tX = torch.tensor(X, device=device); tX2 = torch.tensor(X2, device=device)
    tD = torch.tensor(DOH, device=device)
    tY = torch.tensor(Y, device=device); tW = torch.tensor(W, device=device)

    def epoch_pass(model, opt, idx, train):
        model.train(train)
        total, count = 0.0, 0
        for bi in _batches(len(idx), C.BATCH_SIZE, shuffle=train, rng=rng):
            sel = torch.as_tensor(idx[bi], dtype=torch.long, device=device)
            xb = tX[sel]; x2b = tX2[sel]; yb = tY[sel]; wb = tW[sel]
            
            
            if train and use_date_fuzz:
                doh_np = tD[sel].cpu().numpy()
                doh_np = fuzz_month_onehot(doh_np, fuzz_rng)
                db = torch.tensor(doh_np, device=device)
            else:
                db = tD[sel]
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

    model = DualLegAttnLSTM(use_attention=use_attention).to(device)
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
                print(f"[lstm-attn:{target}] epoch {ep:02d}  train={tr_loss:.4f}  "
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
                        print(f"[lstm-attn:{target}] early stop {ep} (best {best_ep})")
                    break
        else:
            best_ep = ep
    if best_state is not None:
        model.load_state_dict({k: v.to(next(model.parameters()).device) for k, v in best_state.items()})

    if refit and have_val:
        n_ref = best_ep + 1
        trval = np.concatenate([sp.tr, sp.va])
        if verbose:
            print(f"[lstm-attn:{target}] refit on train+val ({len(trval)}) for {n_ref} epochs")
        for ep in range(n_ref):
            tr_loss = epoch_pass(model, opt, trval, True)
            sched.step(tr_loss)
            if verbose:
                print(f"[lstm-attn:{target}] refit epoch {ep:02d}  train={tr_loss:.4f}")

    metrics = {"target": target, "best_epoch": int(best_ep), "best_val_loss": best_val,
               "use_attention": use_attention, "use_date_fuzz": use_date_fuzz}
    return model, metrics


def run_lstm_attn(npz_path, target_1h="z_1h", target_24h="z_24h", device=None,
                  epochs=C.EPOCHS, refit=None, use_date_fuzz=True, use_attention=True,
                  save_dir=None):
    from scorecard import fp_fn_scorecard
    import os
    dev = device if isinstance(device, torch.device) else pick_device(device)
    print(f"[lstm-attn] device={dev.type}  attention={use_attention} date_fuzz={use_date_fuzz}  "
          f"targets=({target_1h},{target_24h})  epochs={epochs}")
    data = dict(np.load(npz_path, allow_pickle=True))
    m1, met1 = train_lstm_attn(data, target_1h, twenty_four=False, device=dev, epochs=epochs,
                               refit=refit, use_date_fuzz=use_date_fuzz, use_attention=use_attention)
    m24, met24 = train_lstm_attn(data, target_24h, twenty_four=True, device=dev, epochs=epochs,
                                 refit=refit, use_date_fuzz=use_date_fuzz, use_attention=use_attention)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        torch.save(m1.state_dict(), os.path.join(save_dir, "model_1h.pt"))
        torch.save(m24.state_dict(), os.path.join(save_dir, "model_24h.pt"))
        print(f"[lstm-attn] saved encoders -> {save_dir}")
    print(f"\n>>> LSTM-attn trained on {target_1h}, scored on z_1h:")
    fp_fn_scorecard(m1, data, target_1h)
    print(f"\n>>> LSTM-attn trained on {target_24h}, scored on z_24h:")
    fp_fn_scorecard(m24, data, target_24h)
    return {"model_1h": m1, "model_24h": m24, "metrics": {target_1h: met1, target_24h: met24}}


if __name__ == "__main__":
    import os
    run_lstm_attn(os.path.join(C.BUILD_DIR, "dataset_multi.npz"),
                  use_date_fuzz=True, use_attention=True, save_dir="model_package_lstm_attn")
