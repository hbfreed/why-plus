"""Fun #4: stuff aging curve. Pitcher birth years via MLB statsapi (no Cloudflare).

    uv run --with matplotlib python experiments/fun_aging.py
"""
import datetime as dt
import json
import urllib.request

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from whyplus.model.data import FULL_START, load_statcast  # noqa: E402
from whyplus.model.features import INPUT_COLS, add_fastball_context, mirror_lhp  # noqa: E402
from xrv import MLP  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(path="artifacts/xrv_model.pt"):
    ck = torch.load(path, weights_only=True)
    m = MLP(len(ck["input_cols"])).to(DEVICE)
    m.load_state_dict(ck["state_dict"]); m.eval()
    return m, np.array(ck["scaler_mean"], "float32"), np.array(ck["scaler_scale"], "float32")


@torch.no_grad()
def predict(m, Xr, mean, scale, bs=500_000):
    Xs = ((Xr - mean) / scale).astype("float32")
    return np.concatenate([m(torch.from_numpy(Xs[i:i+bs]).to(DEVICE)).cpu().numpy()
                           for i in range(0, len(Xs), bs)])


df = add_fastball_context(mirror_lhp(load_statcast(FULL_START, dt.date.today().isoformat())))
df = df[df[INPUT_COLS].notna().all(axis=1)].reset_index(drop=True)
m, mean, scale = load_model()
df["stuff"] = predict(m, df[INPUT_COLS].to_numpy("float32"), mean, scale)

g = df.groupby(["pitcher", "game_year"]).agg(stuff=("stuff", "mean"), n=("stuff", "size")).reset_index()
g = g[g["n"] >= 300].copy()

# birth years via MLB statsapi (batched)
ids = [int(i) for i in g["pitcher"].unique()]
birth = {}
for i in range(0, len(ids), 100):
    chunk = ids[i:i+100]
    url = "https://statsapi.mlb.com/api/v1/people?personIds=" + ",".join(map(str, chunk))
    try:
        data = json.load(urllib.request.urlopen(url, timeout=30))
        for p in data.get("people", []):
            bd = p.get("birthDate")
            if bd:
                birth[p["id"]] = int(bd[:4])
    except Exception as e:
        print(f"statsapi chunk failed: {e}")
print(f"got birth years for {len(birth)}/{len(ids)} pitchers")

g["birth"] = g["pitcher"].map(birth)
g = g.dropna(subset=["birth"])
g["age"] = (g["game_year"] - g["birth"]).astype(int)
g["stuff_z"] = (g["stuff"] - g["stuff"].mean()) / g["stuff"].std()

curve = g[g["age"].between(21, 41)].groupby("age").agg(
    stuff=("stuff_z", "mean"), n=("stuff_z", "size")).reset_index()
print("\n=== stuff (z-score) by age ===")
print(curve.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

cc = curve[curve["n"] >= 10]
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(cc["age"], cc["stuff"], "o-", lw=2.2, color="#d1495b")
ax.axhline(0, color="gray", ls="--", lw=1, label="league average")
ax.set_xlabel("age"); ax.set_ylabel("stuff (z-score, league avg = 0)")
ax.set_title("Stuff aging curve (raw-physics model)\n"
             "caveat: survivorship — only pitchers good enough to last appear at older ages")
ax.grid(alpha=0.25); ax.legend()
fig.tight_layout()
fig.savefig("figures/stuff_aging_curve.png", dpi=120)
print("saved figures/stuff_aging_curve.png")
