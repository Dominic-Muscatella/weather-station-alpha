"""
model.py
========
DualLegConvNet: the architecture you specified.

  * Leg A  -> 7-day hourly series (3 x 168)   + date one-hot
  * Leg B  -> 48-h 15-min series  (3 x 192)   + date one-hot
  Each leg: 1-D conv stack -> global pooling -> concat the date one-hot ->
  per-leg FC.  Both legs merge -> shared FC head with dropout -> 8 logits.

The dropout layers in the head are the Monte-Carlo dropout layers: at inference
we leave them ON for the MC passes and turn them OFF for the raw pass (see
inference.py). Outputs are raw logits; apply sigmoid outside (BCEWithLogitsLoss
during training is numerically stabler).
"""
from __future__ import annotations
import torch
import torch.nn as nn

import config as C
import os


def _disable_mc_dropout(model):
    model.eval()
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.eval()


def _enable_mc_dropout(model):
    model.eval()
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.025
            m.train()


def _conv_block(cin, cout, k):
    pad = k // 2
    return nn.Sequential(
        nn.Conv1d(cin, cout, k, padding=pad),
        nn.BatchNorm1d(cout),
        nn.ReLU(inplace=True),
    )


class ConvLeg(nn.Module):

    def __init__(self, in_ch=C.N_CHANNELS, widths=C.CONV_CHANNELS, k=C.CONV_KERNEL,
                 date_dim=C.DATE_FEAT_DIM, out_dim=C.LEG_FC):
        super().__init__()
        chs = [in_ch] + list(widths)
        self.convs = nn.Sequential(*[_conv_block(chs[i], chs[i + 1], k)
                                     for i in range(len(widths))])
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.gmp = nn.AdaptiveMaxPool1d(1)
        pooled = widths[-1] * 2                      
        self.fc = nn.Sequential(
            nn.Linear(pooled + date_dim, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, series, date_oh):
        
        x = series.transpose(1, 2)
        x = self.convs(x)
        x = torch.cat([self.gap(x).squeeze(-1), self.gmp(x).squeeze(-1)], dim=1)
        x = torch.cat([x, date_oh], dim=1)           
        return self.fc(x)


class DualLegConvNet(nn.Module):
    def __init__(self, n_outputs=C.N_OUTPUTS, head_fc=C.HEAD_FC, p=C.DROPOUT_P):
        super().__init__()
        self.leg_hourly = ConvLeg()                  
        self.leg_sub = ConvLeg()                     
        merged = C.LEG_FC * 2
        layers = []
        dims = [merged] + list(head_fc)
        for i in range(len(head_fc)):
            layers += [nn.Linear(dims[i], dims[i + 1]),
                       nn.ReLU(inplace=True),
                       nn.Dropout(p)]                 
        layers += [nn.Linear(dims[-1], n_outputs)]
        self.head = nn.Sequential(*layers)

    def forward(self, x_hourly, x_sub, date_oh):
        a = self.leg_hourly(x_hourly, date_oh)
        b = self.leg_sub(x_sub, date_oh)
        return self.head(torch.cat([a, b], dim=1))   


def enable_mc_dropout(model: nn.Module):

    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

def load_model(arch, state_path, device):
    if arch == "lstm":
        from engine.models.lstm_experiment import DualLegLSTM as M
        model = M()
    elif arch == "lstm_delta":
        from engine.models.lstm_delta_experiment import DualLegLSTMDelta as M
        model = M()
    elif arch == "lstm_attn":
        from engine.models.lstm_attn_experiment import DualLegAttnLSTM as M
        model = M(use_attention=True)
    else:
        raise ValueError(f"unknown arch {arch!r}")
    file_name = os.path.basename(state_path)       
    file_name_no_ext = os.path.splitext(file_name)[0]
    model.load_state_dict(torch.load(state_path, map_location=device))
    model = model.to(device)
    model.custom_name = file_name_no_ext
    return model