"""Fun #1 (filthiest pitches) + #3 (stuff that doesn't translate). Loads the saved xRV model."""
import datetime as dt

import numpy as np
import pandas as pd
import pybaseball
import torch

from whyplus.model.data import FULL_START, load_statcast
from whyplus.model.features import INPUT_COLS, add_fastball_context, mirror_lhp
from xrv import MLP

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


def names(ids):
    lk = pybaseball.playerid_reverse_lookup([int(i) for i in ids], key_type="mlbam")
    lk["name"] = lk["name_first"].str.title() + " " + lk["name_last"].str.title()
    return dict(zip(lk["key_mlbam"], lk["name"]))


df = add_fastball_context(mirror_lhp(load_statcast(FULL_START, dt.date.today().isoformat())))
df = df[df[INPUT_COLS].notna().all(axis=1)].reset_index(drop=True)
m, mean, scale = load_model()
df["stuff"] = predict(m, df[INPUT_COLS].to_numpy("float32"), mean, scale)  # higher = nastier
f = lambda x: f"{x:.1f}"

# ---- #1 nastiest single pitch from each of the top arms ----
ok = df[df["release_speed"].between(60, 106)]
best = ok.loc[ok.groupby("pitcher")["stuff"].idxmax()]
top = best.nlargest(20, "stuff").copy()
top["name"] = top["pitcher"].map(names(top["pitcher"].unique()))
print("=== FILTHIEST PITCH from each of the 20 nastiest arms (model's view) ===")
print(top[["name", "game_year", "pitch_type", "release_speed", "pfx_x", "pfx_z",
           "release_spin_rate", "stuff"]].to_string(index=False, float_format=f))

# ---- #3 stuff vs actual results ----
g = df.groupby(["pitcher", "game_year"]).agg(
    stuff=("stuff", "mean"), results=("delta_run_exp", lambda s: -s.mean()),
    n=("stuff", "size")).reset_index()
g = g[g["n"] >= 500].copy()
for c in ["stuff", "results"]:
    g[c + "_z"] = (g[c] - g[c].mean()) / g[c].std()
g["gap"] = g["stuff_z"] - g["results_z"]
g["name"] = g["pitcher"].map(names(g["pitcher"].unique()))
cols = ["name", "game_year", "stuff_z", "results_z", "n"]
fz = lambda x: f"{x:+.2f}"
print("\n=== ELITE STUFF, POOR RESULTS (didn't translate -> command/sequencing/luck) ===")
print(g.sort_values("gap", ascending=False).head(12)[cols].to_string(index=False, float_format=fz))
print("\n=== MODEST STUFF, GREAT RESULTS (punched above their stuff) ===")
print(g.sort_values("gap").head(12)[cols].to_string(index=False, float_format=fz))
