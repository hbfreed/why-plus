"""Fun #6: the model's PLATONIC IDEAL PITCH.

Freeze the trained xRV model and gradient-ASCEND the input to maximize predicted stuff.
  unconstrained : let it run free -> does it converge on a real pitch or a monster?
  box-constrained: clip each feature to the 1st-99th real-data percentile -> the nastiest
                   *plausible* pitch. (Still a "Frankenpitch": each dimension maxed
                   independently, because the model doesn't know the features are
                   physically coupled.)

    uv run python experiments/fun_ideal_pitch.py
"""
import datetime as dt

import numpy as np
import pandas as pd
import torch

from whyplus.model.data import FULL_START, RAW_COLS, load_statcast
from whyplus.model.features import (INPUT_COLS, add_fastball_context, mirror_lhp,
                                    vertical_approach_angle)
from xrv import MLP

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ck = torch.load("artifacts/xrv_model.pt", weights_only=True)
m = MLP(len(ck["input_cols"])).to(DEVICE)
m.load_state_dict(ck["state_dict"]); m.eval()
for p in m.parameters():
    p.requires_grad_(False)
mean = np.array(ck["scaler_mean"], "float32")
scale = np.array(ck["scaler_scale"], "float32")

df = add_fastball_context(mirror_lhp(load_statcast(FULL_START, dt.date.today().isoformat())))
df = df[df[INPUT_COLS].notna().all(axis=1)].reset_index(drop=True)
Xr = df[INPUT_COLS].to_numpy("float32")
Xs = ((Xr - mean) / scale).astype("float32")
lo = torch.tensor(np.percentile(Xs, 1, axis=0), device=DEVICE)
hi = torch.tensor(np.percentile(Xs, 99, axis=0), device=DEVICE)
with torch.no_grad():
    best_real = m(torch.from_numpy(Xs).to(DEVICE)).max().item()

fb_idx = [i for i, c in enumerate(INPUT_COLS) if c.startswith("fb_")]


def ascend(constrained, steps=2000, lr=0.03):
    x = torch.zeros(1, len(INPUT_COLS), device=DEVICE, requires_grad=True)
    opt = torch.optim.Adam([x], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        (-m(x)).backward()
        opt.step()
        with torch.no_grad():
            if constrained:
                x.data = torch.min(torch.max(x.data, lo), hi)
            x.data[0, fb_idx] = 0.0  # hold fastball context at league average
    stuff = m(x).item()
    real = x.detach().cpu().numpy()[0] * scale + mean
    return stuff, dict(zip(INPUT_COLS, real))


for name, con in [("UNCONSTRAINED (no limits)", False),
                  ("BOX-CONSTRAINED (1st-99th pctile)", True)]:
    stuff, rd = ascend(con)
    vaa = float(vertical_approach_angle(pd.DataFrame({k: [rd[k]] for k in ["vy0", "vz0", "ay", "az"]}))[0])
    print(f"\n=== {name} ===")
    print(f"  predicted stuff = {stuff:.3f}     (nastiest REAL pitch = {best_real:.3f})")
    for c in RAW_COLS:
        print(f"  {c:22s} {rd[c]:+8.2f}")
    print(f"  -> release_speed {rd['release_speed']:.0f} mph | VAA {vaa:+.1f} deg | "
          f"spin {rd['release_spin_rate']:.0f} | velocity dir vx0={rd['vx0']:+.1f} vz0={rd['vz0']:+.1f}")
