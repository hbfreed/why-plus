"""xRV detour: does a smoother/broader target (expected run value) raise the fit
ceiling AND align better with FanGraphs Stuff+ than whiff / raw run value?

xRV = run value, but each batted ball's ACTUAL result is replaced by its EXPECTED
run value given how it was hit (binned on Statcast's xwOBA-on-contact). Strips out
defense/park/luck -> smoother target. Non-contact pitches keep their (deterministic)
run value.

Compares three targets on identical pitch+fb inputs: fit, reliability, yoy, and the
correlation of the 2024 grade with FanGraphs Stuff+ (/tmp/fg_2024.csv).

    uv run python experiments/xrv.py
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from whyplus.model.data import FULL_START, load_statcast
from whyplus.model.features import INPUT_COLS, add_fastball_context, mirror_lhp

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WHIFF = {"swinging_strike", "swinging_strike_blocked", "foul_tip"}


class MLP(nn.Module):
    def __init__(self, d_in, d_repr=32, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, d_repr), nn.ReLU(), nn.Linear(d_repr, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def build_xrv(df):
    """y_xRV = -(run value with batted-ball outcomes replaced by expected RV)."""
    dre = df["delta_run_exp"].to_numpy("float64").copy()
    bip = (df["description"].to_numpy() == "hit_into_play")
    ew = df["estimated_woba_using_speedangle"].to_numpy("float64")
    valid = bip & np.isfinite(ew)
    # 1D quantile bins of xwOBA-on-contact -> mean actual run value per bin = E[RV | how hit]
    edges = np.quantile(ew[valid], np.linspace(0, 1, 41))
    edges[0] -= 1e-9; edges[-1] += 1e-9
    b = np.digitize(ew[valid], edges[1:-1])
    binmean = pd.Series(dre[valid]).groupby(b).mean()
    dre[valid] = binmean.reindex(b).to_numpy()  # BIP -> expected; non-BIP untouched
    n_smoothed = int(valid.sum())
    print(f"  xRV: smoothed {n_smoothed:,} batted balls via xwOBA bins")
    return (-dre).astype("float32")


def train(X, y, tr, va, task, epochs=30, bs=131072, lr=2e-3, wd=1e-4):
    torch.manual_seed(0)
    m = MLP(X.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.HuberLoss(delta=1.0) if task == "reg" else nn.BCEWithLogitsLoss()
    Xtr = torch.from_numpy(X[tr]).to(DEVICE); ytr = torch.from_numpy(y[tr]).to(DEVICE)
    Xva = torch.from_numpy(X[va]).to(DEVICE); yva = torch.from_numpy(y[va]).to(DEVICE)
    n, best, state, bad = len(Xtr), float("inf"), None, 0
    for _ in range(epochs):
        m.train(); perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, bs):
            idx = perm[i:i+bs]; opt.zero_grad()
            lossf(m(Xtr[idx]), ytr[idx]).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            vl = lossf(m(Xva), yva).item()
        if vl < best - 1e-7:
            best, bad = vl, 0
            state = {k: v.detach().clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= 5:
                break
    m.load_state_dict(state)
    return m


@torch.no_grad()
def pred(m, X, bs=500_000):
    m.eval()
    return np.concatenate([m(torch.from_numpy(X[i:i+bs]).to(DEVICE)).cpu().numpy()
                           for i in range(0, len(X), bs)])


def reliability(ids, p, min_n=100):
    rng = np.random.default_rng(0)
    d = ids.copy(); d["p"] = p; d["h"] = rng.integers(0, 2, len(p))
    cnt = d.groupby(["pitcher", "game_year"])["p"].transform("size"); d = d[cnt >= min_n]
    piv = d.groupby(["pitcher", "game_year", "h"])["p"].mean().unstack("h").dropna()
    return float(np.corrcoef(piv[0], piv[1])[0, 1]) if len(piv) >= 10 else float("nan")


def yoy(ids, p, rv, min_n=200):
    d = ids.copy(); d["m"] = p; d["rv"] = rv
    g = d.groupby(["pitcher", "game_year"]).agg(m=("m", "mean"), rv=("rv", "mean"),
                                                n=("rv", "size")).reset_index()
    g = g[g.n >= min_n]; nx = g[["pitcher", "game_year", "rv"]].copy(); nx["game_year"] -= 1
    mm = g.merge(nx, on=["pitcher", "game_year"], suffixes=("", "_n"))
    return float(np.corrcoef(mm["m"], mm["rv_n"])[0, 1]) if len(mm) >= 10 else float("nan")


def main():
    print(f"Device: {DEVICE}")
    df = add_fastball_context(mirror_lhp(load_statcast(FULL_START, dt.date.today().isoformat())))
    df = df[df[INPUT_COLS].notna().all(axis=1)].reset_index(drop=True)
    rv = (-df["delta_run_exp"]).to_numpy("float32")
    ids = df[["pitcher", "game_year"]].reset_index(drop=True)
    X = df[INPUT_COLS].to_numpy("float32")
    rng = np.random.default_rng(0)
    va = np.zeros(len(df), bool); va[rng.choice(len(df), int(0.15*len(df)), replace=False)] = True
    tr = ~va
    Xs = StandardScaler().fit(X[tr]).transform(X).astype("float32")
    fg = pd.read_csv("experiments/fg_stuff_2024.csv")
    d24 = (df["game_year"] == 2024).to_numpy()

    targets = {
        "run_value": ("reg", rv),
        "xRV": ("reg", build_xrv(df)),
        "whiff": ("clf", df["description"].isin(WHIFF).to_numpy("float32")),
    }
    print(f"\n{len(df):,} pitches\n")
    print(f"{'target':10s} {'fit':>11s} {'reliab':>7s} {'yoy':>7s} {'vs FanGraphs Stuff+':>22s}")
    for name, (task, y) in targets.items():
        m = train(Xs, y, tr, va, task)
        pa = pred(m, Xs)
        if task == "clf":
            grade = 1 / (1 + np.exp(-np.clip(pa, -30, 30)))
            fit = f"AUC {roc_auc_score(y[va], pa[va]):.3f}"
        else:
            ssr = ((y[va]-pa[va])**2).sum(); sst = ((y[va]-y[va].mean())**2).sum()
            grade = pa
            fit = f"R2 {1-ssr/sst:.3f}"
        rel = reliability(ids, grade)
        yy = yoy(ids, grade, rv)
        g24 = pd.DataFrame({"pitcher": ids["pitcher"][d24], "g": grade[d24]})
        g24 = g24.groupby("pitcher")["g"].agg(["mean", "size"]).reset_index()
        g24 = g24[g24["size"] >= 300]
        mrg = g24.merge(fg, left_on="pitcher", right_on="mlbam")
        fgc = float(np.corrcoef(mrg["mean"], mrg["sp_stuff"])[0, 1])
        fgs = float(spearmanr(mrg["mean"], mrg["sp_stuff"]).correlation)
        print(f"{name:10s} {fit:>11s} {rel:>7.3f} {yy:>+7.3f}   r={fgc:+.2f} ρ={fgs:+.2f} (n={len(mrg)})")


if __name__ == "__main__":
    main()
