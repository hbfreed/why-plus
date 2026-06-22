"""Train the chosen model (xRV target, pitch+fb inputs, d_repr=32) and save a
checkpoint to artifacts/xrv_model.pt (state_dict + scaler + cols + config).

The binary lives in artifacts/ (gitignored); it's fully regenerable from this script.

    uv run python experiments/save_xrv_model.py
"""
import datetime as dt
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from whyplus.model.data import FULL_START, load_statcast
from whyplus.model.features import INPUT_COLS, add_fastball_context, mirror_lhp
from xrv import MLP, build_xrv, train  # reuse the xRV machinery

OUT = Path(__file__).resolve().parents[1] / "artifacts" / "xrv_model.pt"

df = add_fastball_context(mirror_lhp(load_statcast(FULL_START, dt.date.today().isoformat())))
df = df[df[INPUT_COLS].notna().all(axis=1)].reset_index(drop=True)
y = build_xrv(df)
X = df[INPUT_COLS].to_numpy("float32")
rng = np.random.default_rng(0)
va = np.zeros(len(df), bool); va[rng.choice(len(df), int(0.15 * len(df)), replace=False)] = True
sc = StandardScaler().fit(X[~va])
Xs = sc.transform(X).astype("float32")
model = train(Xs, y, ~va, va, "reg")

OUT.parent.mkdir(parents=True, exist_ok=True)
torch.save({
    "state_dict": model.state_dict(),
    "scaler_mean": sc.mean_.astype("float32").tolist(),
    "scaler_scale": sc.scale_.astype("float32").tolist(),
    "input_cols": INPUT_COLS,
    "d_repr": 32,
    "target": "xRV",
    "arch": "input->256->128->32->1 ReLU (penultimate d=32)",
}, OUT)
print(f"saved chosen xRV model -> {OUT}")
