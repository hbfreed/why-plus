"""Compact sweep for the Stuff+/whiff model, RANKED BY STAT QUALITY (yoy + split-half
reliability), not per-pitch fit. Optimizing AUC would tune toward a worse stat.

NO constructed input features anywhere (no diffs, movement, tunnels) - the model must
learn all structure from RAW physics; that is the spine of the project. We sweep:
  inputs   : pitch (12 raw constants only) vs pitch+fb (+ fastball-context means)
  target   : whiff (per pitch) / csw (called+swinging) / whiff-per-swing
  arch/reg : d_repr {16,32,64} x dropout {0,0.2} x weight_decay {1e-5,1e-3}  (AdamW)
Depth is fixed (low-ROI on tabular). Selection is on yoy (across pitcher-seasons, so
it can't be gamed by overfitting one val split).

    uv run python experiments/sweep.py
"""

from __future__ import annotations

import datetime as dt
import itertools

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from whyplus.model.data import FULL_START, RAW_COLS, load_statcast
from whyplus.model.features import INPUT_COLS, add_fastball_context, mirror_lhp

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WHIFF = {"swinging_strike", "swinging_strike_blocked", "foul_tip"}
SWING = WHIFF | {"foul", "hit_into_play"}
CSW = WHIFF | {"called_strike"}

D_REPRS = [16, 32, 64]
HIDDEN = (256, 128)        # depth is low-ROI on tabular; fixed to spend budget elsewhere
DROPOUTS = [0.0, 0.2]
WDS = [1e-5, 1e-3]


class FlexMLP(nn.Module):
    def __init__(self, d_in, hidden, d_repr, dropout):
        super().__init__()
        layers, prev = [], d_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers += [nn.Linear(prev, d_repr), nn.ReLU()]
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(d_repr, 1)

    def forward(self, x):
        p = self.backbone(x)
        return self.head(p).squeeze(-1), p


def train(Xtr, ytr, Xva, yva, *, hidden, d_repr, dropout, wd,
          lr=2e-3, epochs=40, bs=131072, patience=5, seed=0):
    torch.manual_seed(seed)
    model = FlexMLP(Xtr.shape[1], hidden, d_repr, dropout).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.BCEWithLogitsLoss()
    n = len(Xtr)
    best, best_state, bad = float("inf"), None, 0
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            opt.zero_grad()
            loss_fn(model(Xtr[idx])[0], ytr[idx]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = loss_fn(model(Xva)[0], yva).item()
        if vl < best - 1e-7:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict_prob(model, Xg, idx, bs=500_000):
    model.eval()
    X = Xg[idx]
    out = []
    for i in range(0, len(X), bs):
        out.append(torch.sigmoid(model(X[i : i + bs])[0]).cpu().numpy())
    return np.concatenate(out)


def reliability(ids, pred, min_n=100, seed=0):
    rng = np.random.default_rng(seed)
    d = ids.copy()
    d["pred"], d["half"] = pred, rng.integers(0, 2, len(pred))
    cnt = d.groupby(["pitcher", "game_year"])["pred"].transform("size")
    d = d[cnt >= min_n]
    piv = d.groupby(["pitcher", "game_year", "half"])["pred"].mean().unstack("half").dropna()
    return float(np.corrcoef(piv[0], piv[1])[0, 1]) if len(piv) >= 10 else float("nan")


def yoy(ids, pred, rv, min_n=200):
    d = ids.copy()
    d["metric"], d["rv"] = pred, rv
    g = d.groupby(["pitcher", "game_year"]).agg(
        metric=("metric", "mean"), rv=("rv", "mean"), n=("rv", "size")).reset_index()
    g = g[g.n >= min_n]
    nxt = g[["pitcher", "game_year", "rv"]].copy()
    nxt["game_year"] -= 1
    m = g.merge(nxt, on=["pitcher", "game_year"], suffixes=("", "_next"))
    if len(m) < 10:
        return float("nan"), float("nan")
    return (float(np.corrcoef(m["metric"], m["rv_next"])[0, 1]),
            float(np.corrcoef(m["rv"], m["rv_next"])[0, 1]))


def main():
    print(f"Device: {DEVICE}")
    df = load_statcast(FULL_START, dt.date.today().isoformat())
    df = mirror_lhp(df)
    df = add_fastball_context(df)
    df = df[df[INPUT_COLS].notna().all(axis=1)].reset_index(drop=True)
    df["run_value"] = (-df["delta_run_exp"]).astype("float32")
    desc = df["description"]
    is_whiff = desc.isin(WHIFF).to_numpy()
    is_csw = desc.isin(CSW).to_numpy()
    is_swing = desc.isin(SWING).to_numpy()
    rv = df["run_value"].to_numpy("float32")
    ids_all = df[["pitcher", "game_year"]].reset_index(drop=True)

    rng = np.random.default_rng(0)
    val_all = np.zeros(len(df), bool)
    val_all[rng.choice(len(df), int(0.15 * len(df)), replace=False)] = True

    # Input-purity axis. NO constructed features in either set (no diffs/movement/
    # tunnels). 'pitch' = the pitch's own raw constants; 'pitch+fb' adds only the raw-
    # constant MEANS of the pitcher's primary fastball as context (the model still has
    # to learn any differential itself).
    input_sets = {"pitch": RAW_COLS, "pitch+fb": INPUT_COLS}
    Xg = {}
    for sname, cols in input_sets.items():
        X = df[cols].to_numpy("float32")
        sc = StandardScaler().fit(X[~val_all])
        Xg[sname] = torch.from_numpy(sc.transform(X).astype("float32")).to(DEVICE)

    print(f"{len(df):,} pitches | whiff {is_whiff.mean():.3f} csw {is_csw.mean():.3f} "
          f"swing {is_swing.mean():.3f}\n")

    targets = {
        "whiff": (np.ones(len(df), bool), is_whiff.astype("float32")),
        "csw": (np.ones(len(df), bool), is_csw.astype("float32")),
        "whiff/swing": (is_swing, is_whiff.astype("float32")),
    }

    results = []
    for tname, (rowmask, label) in targets.items():
        rows = np.where(rowmask)[0]
        y = label[rows]
        val = val_all[rows]
        ids_t = ids_all.iloc[rows].reset_index(drop=True)
        rv_t = rv[rows]
        rows_t = torch.tensor(rows, device=DEVICE)
        tr_idx = torch.tensor(rows[~val], device=DEVICE)
        va_idx = torch.tensor(rows[val], device=DEVICE)
        ytr = torch.tensor(y[~val], device=DEVICE)
        yva = torch.tensor(y[val], device=DEVICE)
        _, base = yoy(ids_t, np.zeros(len(rows)), rv_t)
        print(f"-- target={tname} (n={len(rows):,}, base={y.mean():.3f}, yoy_baseline={base:+.3f})")
        for sname in input_sets:
            Xtr, Xva = Xg[sname][tr_idx], Xg[sname][va_idx]
            for d_repr, dropout, wd in itertools.product(D_REPRS, DROPOUTS, WDS):
                model = train(Xtr, ytr, Xva, yva, hidden=HIDDEN, d_repr=d_repr, dropout=dropout, wd=wd)
                pall = predict_prob(model, Xg[sname], rows_t)
                au = roc_auc_score(y[val], pall[val])
                rel = reliability(ids_t, pall)
                ym, _ = yoy(ids_t, pall, rv_t)
                results.append((tname, sname, d_repr, dropout, wd, au, rel, ym, base))
                print(f"   in={sname:8s} d_repr={d_repr:2d} drop={dropout} wd={wd:.0e}  "
                      f"AUC={au:.3f} reliab={rel:.3f} yoy={ym:+.3f}")

    print("\n=== TOP 12 BY yoy (then reliability) ===")
    print(f"{'target':12s} {'inputs':9s} {'d_repr':>6s} {'drop':>5s} {'wd':>6s} "
          f"{'AUC':>6s} {'reliab':>7s} {'yoy':>7s} {'base':>7s}")
    ranked = sorted(results, key=lambda r: (-(r[7] if r[7] == r[7] else -9),
                                            -(r[6] if r[6] == r[6] else -9)))
    for t, sn, dr, dp, wd, au, rel, ym, base in ranked[:12]:
        print(f"{t:12s} {sn:9s} {dr:>6d} {dp:>5} {wd:>6.0e} "
              f"{au:>6.3f} {rel:>7.3f} {ym:>+7.3f} {base:>+7.3f}")


if __name__ == "__main__":
    main()
