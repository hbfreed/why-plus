"""Showdown: does a Ridge regression do what the net does? (xRV target, same split)

  ridge (linear) : a plain weighted sum of the 24 raw features. No nonlinearity.
  ridge + poly2  : + squares + pairwise interactions -> hands the regression the
                   nonlinearity the net finds for free. (Diagnostic only: intentionally
                   breaks the no-constructed-features rule to isolate the net's edge.)
  net (MLP)      : the deep model.

Compared on R2, reliability, yoy, and correlation with FanGraphs Stuff+.

    uv run python experiments/showdown.py
"""
import datetime as dt

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from whyplus.model.data import FULL_START, load_statcast
from whyplus.model.features import INPUT_COLS, add_fastball_context, mirror_lhp
from xrv import MLP, build_xrv

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

df = add_fastball_context(mirror_lhp(load_statcast(FULL_START, dt.date.today().isoformat())))
df = df[df[INPUT_COLS].notna().all(axis=1)].reset_index(drop=True)
y = build_xrv(df)
ids = df[["pitcher", "game_year"]].reset_index(drop=True)
rv = (-df["delta_run_exp"]).to_numpy("float32")
X = df[INPUT_COLS].to_numpy("float32")
rng = np.random.default_rng(0)
va = np.zeros(len(df), bool); va[rng.choice(len(df), int(0.15 * len(df)), replace=False)] = True
tr = ~va
yv = y[va]
fg = pd.read_csv("experiments/fg_stuff_2024.csv")
d24 = df["game_year"].to_numpy() == 2024
sc = StandardScaler().fit(X[tr])
Xs = sc.transform(X).astype("float32")


def reliability(p, min_n=100):
    r = np.random.default_rng(0)
    d = ids.copy(); d["p"] = p; d["h"] = r.integers(0, 2, len(p))
    cnt = d.groupby(["pitcher", "game_year"])["p"].transform("size"); d = d[cnt >= min_n]
    piv = d.groupby(["pitcher", "game_year", "h"])["p"].mean().unstack("h").dropna()
    return float(np.corrcoef(piv[0], piv[1])[0, 1]) if len(piv) >= 10 else float("nan")


def yoy(p, min_n=200):
    d = ids.copy(); d["m"] = p; d["rv"] = rv
    g = d.groupby(["pitcher", "game_year"]).agg(m=("m", "mean"), rv=("rv", "mean"), n=("rv", "size")).reset_index()
    g = g[g.n >= min_n]; nx = g[["pitcher", "game_year", "rv"]].copy(); nx["game_year"] -= 1
    mm = g.merge(nx, on=["pitcher", "game_year"], suffixes=("", "_n"))
    return float(np.corrcoef(mm["m"], mm["rv_n"])[0, 1]) if len(mm) >= 10 else float("nan")


def fgcorr(p):
    g = pd.DataFrame({"pitcher": ids["pitcher"][d24], "g": p[d24]}).groupby("pitcher")["g"].agg(
        ["mean", "size"]).reset_index()
    g = g[g["size"] >= 300]
    mrg = g.merge(fg, left_on="pitcher", right_on="mlbam")
    return float(np.corrcoef(mrg["mean"], mrg["sp_stuff"])[0, 1])


def report(name, pva, pall):
    r2 = 1 - ((yv - pva) ** 2).sum() / ((yv - yv.mean()) ** 2).sum()
    print(f"  {name:16s} R2={r2:+.4f}   reliab={reliability(pall):.3f}   "
          f"yoy={yoy(pall):+.3f}   FanGraphs-Stuff+ r={fgcorr(pall):+.2f}")


print("xRV target, identical split — three models:\n")

# 1. ridge (linear)
rg = Ridge(alpha=10.0).fit(Xs[tr], y[tr])
report("ridge (linear)", rg.predict(Xs[va]), rg.predict(Xs))

# 2. ridge + squares + interactions (fit on a 1M sample, predict in chunks)
poly = PolynomialFeatures(2, include_bias=False)
samp = rng.choice(np.where(tr)[0], 1_000_000, replace=False)
rg2 = Ridge(alpha=50.0).fit(poly.fit_transform(Xs[samp]), y[samp])
pp = lambda Z: np.concatenate([rg2.predict(poly.transform(Z[i:i+300000])) for i in range(0, len(Z), 300000)])
report("ridge + poly2", pp(Xs[va]), pp(Xs))

# 3. the net
torch.manual_seed(0)
m = MLP(Xs.shape[1]).to(DEVICE)
opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-4)
lossf = nn.HuberLoss(delta=1.0)
Xtr = torch.from_numpy(Xs[tr]).to(DEVICE); ytr = torch.from_numpy(y[tr]).to(DEVICE)
Xva = torch.from_numpy(Xs[va]).to(DEVICE); yva = torch.from_numpy(yv).to(DEVICE)
n, best, state, bad = len(Xtr), 1e9, None, 0
for _ in range(25):
    m.train(); perm = torch.randperm(n, device=DEVICE)
    for i in range(0, n, 131072):
        idx = perm[i:i+131072]; opt.zero_grad()
        lossf(m(Xtr[idx]), ytr[idx]).backward(); opt.step()
    m.eval()
    with torch.no_grad():
        vl = lossf(m(Xva), yva).item()
    if vl < best - 1e-7:
        best, bad = vl, 0; state = {k: v.detach().clone() for k, v in m.state_dict().items()}
    else:
        bad += 1
        if bad >= 5:
            break
m.load_state_dict(state); m.eval()
with torch.no_grad():
    pva = m(Xva).cpu().numpy()
    pall = np.concatenate([m(torch.from_numpy(Xs[i:i+500000]).to(DEVICE)).cpu().numpy()
                           for i in range(0, len(Xs), 500000)])
report("net (MLP)", pva, pall)
