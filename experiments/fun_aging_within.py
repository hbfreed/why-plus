"""Fun #4b: WITHIN-pitcher (delta-method) stuff aging curve, two ways.

For each pitcher in consecutive seasons, take the CHANGE in stuff (age Y -> Y+1),
average by age, and chain into a curve. Every pitcher is his own control, so this
removes between-pitcher survivorship.

We compute it two ways for the z-score baseline:
  - pooled    : z-scored across ALL pitcher-seasons  -> the 2021-26 league-wide stuff
                inflation (velo + sweepers) leaks into the age axis (era confound).
  - per-season: z-scored WITHIN each season          -> measures age vs the league THAT
                year, removing the era trend. This is the honest age curve.
(Residual caveat both ways: the delta method still mildly over-states late-career stuff,
since the pitchers who decline most get cut and stop contributing deltas.)

    uv run --with matplotlib python experiments/fun_aging_within.py
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
g = g[g["n"] >= 200].copy()
g["sz_pool"] = (g["stuff"] - g["stuff"].mean()) / g["stuff"].std()
g["sz_seas"] = g.groupby("game_year")["stuff"].transform(lambda s: (s - s.mean()) / s.std())

# birth years via MLB statsapi
ids = [int(i) for i in g["pitcher"].unique()]
birth = {}
for i in range(0, len(ids), 100):
    url = "https://statsapi.mlb.com/api/v1/people?personIds=" + ",".join(map(str, ids[i:i+100]))
    try:
        for p in json.load(urllib.request.urlopen(url, timeout=30)).get("people", []):
            if p.get("birthDate"):
                birth[p["id"]] = int(p["birthDate"][:4])
    except Exception as e:
        print(f"statsapi chunk failed: {e}")
g["age"] = g["game_year"] - g["pitcher"].map(birth)
g = g.dropna(subset=["age"]); g["age"] = g["age"].astype(int)


def delta_curve(col):
    nxt = g[["pitcher", "game_year", col]].rename(columns={col: "next"})
    nxt["game_year"] = nxt["game_year"] - 1
    pr = g.merge(nxt, on=["pitcher", "game_year"])
    pr["delta"] = pr["next"] - pr[col]
    a = pr.groupby("age").agg(dmean=("delta", "mean"), npairs=("delta", "size")).reset_index()
    a = a[(a["npairs"] >= 20) & a["age"].between(22, 37)].sort_values("age")
    ages = list(a["age"])
    curve = {ages[0]: 0.0}
    for ag in ages:
        curve[ag + 1] = curve.get(ag, 0.0) + float(a.loc[a["age"] == ag, "dmean"].iloc[0])
    cur = pd.Series(curve).sort_index()
    return a, cur - cur.max()


a_pool, cur_pool = delta_curve("sz_pool")
a_seas, cur_seas = delta_curve("sz_seas")

print("=== Δstuff/yr by age (per-season-detrended = honest age signal) ===")
print("  age   pooled   per-season   n_pairs")
for _, r in a_seas.iterrows():
    dp = a_pool.loc[a_pool["age"] == r["age"], "dmean"]
    print(f"  {int(r.age):>3d}   {float(dp.iloc[0]) if len(dp) else float('nan'):+.3f}    "
          f"{r.dmean:+.3f}       {int(r.npairs)}")
print(f"\nper-season curve peak age = {cur_seas.idxmax()}")
print(cur_seas.round(2).to_string())

fig, ax = plt.subplots(figsize=(9.5, 5.2))
ax.plot(cur_pool.index, cur_pool.values, "o--", lw=1.8, color="#bbbbbb",
        label="pooled (era trend leaks in)")
ax.plot(cur_seas.index, cur_seas.values, "o-", lw=2.6, color="#2a9d8f",
        label="per-season de-trended (honest age curve)")
ax.axhline(0, color="gray", ls=":", lw=1)
ax.set_xlabel("age"); ax.set_ylabel("stuff vs peak (league SD units)")
ax.set_title("Within-pitcher stuff aging curve (delta method)\n"
             "survivorship removed; per-season line also removes the 2021–26 era inflation")
ax.grid(alpha=0.25); ax.legend()
fig.tight_layout()
fig.savefig("figures/stuff_aging_within.png", dpi=120)
print("\nsaved figures/stuff_aging_within.png")
