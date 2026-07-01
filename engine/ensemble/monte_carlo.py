from engine.inference import normalize
import copy
import numpy as np
import torch
import config as C
from engine.model import _enable_mc_dropout



def compute_random_transition(v_minus_1, future_val, suddenness_factor=4.5, scale=0.05):

    channel_averages = (v_minus_1 + future_val) / 2.0
    dynamic_scales = np.where(channel_averages > 150, scale / 10.0, scale)

    suddenness_factor = np.random.normal(loc=suddenness_factor, scale=2.25)
    suddenness_factor = np.clip(suddenness_factor, 2.5, 9.0)

    raw_shift = np.random.normal(loc=0.0, scale=suddenness_factor * 0.45)
    max_safe_shift = suddenness_factor * 0.65
    center_shift = np.clip(raw_shift, -max_safe_shift, max_safe_shift)

    t = np.linspace(-suddenness_factor, suddenness_factor, 12)
    t_shifted = t + center_shift
    sigmoid_profile = (1 / (1 + np.exp(-t_shifted)))[:, np.newaxis]
    interp_steps = v_minus_1 + (future_val - v_minus_1) * sigmoid_profile

    interp_noise = np.ones((12, 3))
    interp_noise[1:-1, :] = np.random.normal(loc=1.0, scale=dynamic_scales, size=(10, 3))

    interp_noise[1, :] = (interp_noise[1, :] + 2.0) / 3.0
    interp_noise[2, :] = (interp_noise[2, :] + 1.0) / 2.0
    interp_noise[3, :] = (interp_noise[3, :] + 0.333) / 1.333

    interp_noise[-4, :] = (interp_noise[-4, :] + 0.333) / 1.333
    interp_noise[-3, :] = (interp_noise[-3, :] + 1.0) / 2.0
    interp_noise[-2, :] = (interp_noise[-2, :] + 2.0) / 3.0

    return interp_steps * interp_noise


@torch.no_grad()
def mc_predict(model, xH, xS, doh, scaler_mean, scaler_std, device, passes=50):

    _enable_mc_dropout(model)
    xh = copy.deepcopy(xH)
    xs = copy.deepcopy(xS)
    
    v_minus_2 = (xh[-3] + xh[-4]) / 2  
    v_minus_1 = (xh[-1] + xh[-2]) / 2  
    
    denom = np.where(v_minus_2 == 0, 1e-6, v_minus_2)
    pct_change = ((v_minus_1 - v_minus_2) / denom)/1.5
    accumulated_batches = []
    
    projected_mean = v_minus_1 * (1.0 + pct_change)
    
    projected_std = np.maximum(np.abs(projected_mean * pct_change), 1)
    projected_std = np.maximum(projected_std, 1e-4)  
    for batch in range(C.MC_BATCHES):
        print(f"[model run] batch {batch+1}/{C.MC_BATCHES}, {model.custom_name}")
        
        future_anchors = np.random.normal(loc=projected_mean, scale=projected_std, size=(passes, 3))
        
        
        Xh_batch = np.zeros((passes, C.HOURLY_WINDOW_LEN, len(C.CHANNELS)), dtype=np.float32)
        Xs_batch = np.zeros((passes, 576, len(C.CHANNELS)), dtype=np.float32)
        
        
        for i in range(passes):
            future_val = future_anchors[i]

            shifted_h = np.empty_like(xh)
            shifted_h[:-1] = xh[1:]
            shifted_h[-1] = future_val
            Xh_batch[i] = shifted_h

            shifted_s = np.empty_like(xs)
            shifted_s[:-12] = xs[12:]
            shifted_s[-12:] = compute_random_transition(v_minus_1, future_val, scale=0.05)
            Xs_batch[i] = shifted_s

        Xh_batch = normalize(Xh_batch, scaler_mean, scaler_std).astype(np.float32)
        Xs_batch = normalize(Xs_batch, scaler_mean, scaler_std).astype(np.float32)

        td = torch.tensor(np.repeat(doh[None], passes, axis=0), device=device)
        th = torch.tensor(Xh_batch, device=device)
        ts = torch.tensor(Xs_batch, device=device)

        logits = model(th, ts, td)
        probs = torch.sigmoid(logits).cpu().numpy()
        accumulated_batches.append(probs)

    print("[model run] concatenating probs...")
    all_raw_probs = np.concatenate(accumulated_batches, axis=0)
    print("[model run] calculating mean...")
    final_probs = all_raw_probs.mean(axis=0)
    print("[model run] getting percentiles...")
    final_low, final_high = np.percentile(all_raw_probs, getattr(C, "MC_CI", (15, 85)), axis=0)
    return final_probs, final_low, final_high