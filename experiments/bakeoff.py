"""Bake-off: is Pitching+ or a learnable-target Stuff+ a better EXPLAINABLE pitch
model than the brief's run-value Stuff+?

2x2 design:  inputs {stuff, pitching}  x  target {run_value (Huber), whiff (BCE)}.

We judge each cell the way you judge a real pitch-quality stat - NOT by per-pitch
accuracy, but by whether it measures a repeatable skill:
  - fit          : per-pitch R^2 (run_value) or AUC (whiff)     <- expected weak
  - active dims  : live representation dims out of 32           <- explainability substrate
  - reliability  : split-half correlation of the aggregated rating  <- is it a GOOD STAT?
  - validity     : corr(season metric, season actual run prevention)
  - yoy          : corr(season metric, NEXT season run prevention)  <- the gold standard,
                   beat the 'past results' baseline to be a real leading indicator.

Training keeps the whole dataset resident on the GPU and minibatches by index, so
each ~80k-param fit takes seconds (no per-batch host->device copy).

    uv run python experiments/bakeoff.py
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from whyplus.model.data import FULL_START, load_statcast
from whyplus.model.features import INPUT_COLS, add_fastball_context, mirror_lhp
from whyplus.model.net import StuffMLP

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WHIFF = {"swinging_strike", "swinging_strike_blocked", "foul_tip"}
PITCH_EXTRA = ["plate_x", "plate_z", "balls", "strikes", "platoon_same"]

# (label, input set, task)
CONFIGS = [
    ("Stuff+ / runval", "stuff", "reg"),
    ("Pitching+ / runval", "pitching", "reg"),
    ("Stuff+ / whiff", "stuff", "clf"),
    ("Pitching+ / whiff", "pitching", "clf"),
]


def build_frame() -> pd.DataFrame:
    df = load_statcast(FULL_START, dt.date.today().isoformat())
    df = mirror_lhp(df)             # mirrors physics + pfx_x + plate_x
    df = add_fastball_context(df)
    df["run_value"] = (-df["delta_run_exp"]).astype("float32")
    df["whiff"] = df["description"].isin(WHIFF).astype("float32")
    df["platoon_same"] = (df["p_throws"].to_numpy() == df["stand"].to_numpy()).astype("float32")
    # one common valid-row mask so every cell trains on identical rows
    need = INPUT_COLS + ["plate_x", "plate_z", "balls", "strikes"]
    mask = df[need].notna().all(axis=1).to_numpy()
    df = df[mask].reset_index(drop=True)
    return df


def cols_for(kind: str) -> list[str]:
    return INPUT_COLS if kind == "stuff" else INPUT_COLS + PITCH_EXTRA


def train_cell(X, y, tr, va, task, *, seed=0, epochs=50, bs=65536, lr=1e-3, wd=1e-5, patience=6):
    torch.manual_seed(seed)
    model = StuffMLP(d_in=X.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.HuberLoss(delta=1.0) if task == "reg" else nn.BCEWithLogitsLoss()
    Xtr = torch.from_numpy(X[tr]).to(DEVICE)
    ytr = torch.from_numpy(y[tr]).to(DEVICE)
    Xva = torch.from_numpy(X[va]).to(DEVICE)
    yva = torch.from_numpy(y[va]).to(DEVICE)
    n = len(Xtr)
    best, best_state, bad = float("inf"), None, 0
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            opt.zero_grad()
            p, _ = model(Xtr[idx])
            loss_fn(p, ytr[idx]).backward()
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
def infer(model, X, want_rep=False, bs=300_000):
    model.eval()
    preds, reps = [], []
    Xt = torch.from_numpy(X)
    for i in range(0, len(Xt), bs):
        p, r = model(Xt[i : i + bs].to(DEVICE))
        preds.append(p.cpu().numpy())
        if want_rep:
            reps.append(r.cpu().numpy())
    return np.concatenate(preds), (np.concatenate(reps) if want_rep else None)


def reliability(ids, pred, min_n=100, seed=0):
    """Split-half: split each pitcher-season's pitches in two, aggregate the rating
    on each half, correlate across pitcher-seasons. High = stable skill metric."""
    rng = np.random.default_rng(seed)
    d = ids[["pitcher", "game_year"]].copy()
    d["pred"] = pred
    d["half"] = rng.integers(0, 2, len(d))
    cnt = d.groupby(["pitcher", "game_year"])["pred"].transform("size")
    d = d[cnt >= min_n]
    piv = d.groupby(["pitcher", "game_year", "half"])["pred"].mean().unstack("half").dropna()
    if len(piv) < 10:
        return float("nan"), len(piv)
    return float(np.corrcoef(piv[0], piv[1])[0, 1]), len(piv)


def _season_agg(ids, pred, rv, min_n):
    d = ids[["pitcher", "game_year"]].copy()
    d["metric"], d["rv"] = pred, rv
    g = d.groupby(["pitcher", "game_year"]).agg(
        metric=("metric", "mean"), rv=("rv", "mean"), n=("rv", "size")
    ).reset_index()
    return g[g.n >= min_n]


def construct_validity(ids, pred, rv, min_n=100):
    g = _season_agg(ids, pred, rv, min_n)
    return float(np.corrcoef(g["metric"], g["rv"])[0, 1]), len(g)


def year_over_year(ids, pred, rv, min_n=200):
    """corr(season metric, NEXT season actual run prevention) vs the past-results
    baseline corr(season rv, next season rv)."""
    g = _season_agg(ids, pred, rv, min_n)
    nxt = g[["pitcher", "game_year", "rv"]].copy()
    nxt["game_year"] -= 1
    m = g.merge(nxt, on=["pitcher", "game_year"], suffixes=("", "_next"))
    if len(m) < 10:
        return float("nan"), float("nan"), len(m)
    r_metric = float(np.corrcoef(m["metric"], m["rv_next"])[0, 1])
    r_past = float(np.corrcoef(m["rv"], m["rv_next"])[0, 1])
    return r_metric, r_past, len(m)


def main():
    print(f"Device: {DEVICE}")
    df = build_frame()
    rv = df["run_value"].to_numpy(np.float32)
    ids = df[["pitcher", "game_year"]]
    n = len(df)
    rng = np.random.default_rng(0)
    va_mask = np.zeros(n, bool)
    va_mask[rng.choice(n, int(0.15 * n), replace=False)] = True
    tr, va = np.where(~va_mask)[0], np.where(va_mask)[0]
    print(f"{n:,} pitches | whiff base rate {df['whiff'].mean():.3f} | run_value std {rv.std():.4f}\n")

    rows = []
    for name, kind, task in CONFIGS:
        X = df[cols_for(kind)].to_numpy(np.float32)
        scaler = StandardScaler().fit(X[tr])
        Xs = scaler.transform(X).astype(np.float32)
        y = (df["run_value"] if task == "reg" else df["whiff"]).to_numpy(np.float32)

        model = train_cell(Xs, y, tr, va, task)
        pred_va, rep_va = infer(model, Xs[va], want_rep=True)
        pred_all, _ = infer(model, Xs)
        if task == "clf":
            pred_all = 1.0 / (1.0 + np.exp(-np.clip(pred_all, -30, 30)))  # prob to aggregate
            fit, fitname = roc_auc_score(y[va], pred_va), "AUC"
        else:
            ssr = float(((y[va] - pred_va) ** 2).sum())
            sst = float(((y[va] - y[va].mean()) ** 2).sum())
            fit, fitname = 1 - ssr / sst, "R2 "
        active = int((rep_va.std(0) > 1e-4).sum())
        rel, _ = reliability(ids, pred_all)
        cv, _ = construct_validity(ids, pred_all, rv)
        yoy_m, yoy_p, n_yoy = year_over_year(ids, pred_all, rv)
        rows.append((name, fitname, fit, active, rel, cv, yoy_m, yoy_p, n_yoy))
        print(f"  {name:20s} {fitname}={fit:+.3f}  active={active:2d}/32  "
              f"reliab={rel:.3f}  valid={cv:+.3f}  yoy={yoy_m:+.3f}")

    print("\n=== BAKE-OFF SUMMARY (the stat-quality columns are what matter) ===")
    print(f"{'config':20s} {'fit':>9s} {'active':>7s} {'reliab':>7s} {'valid':>7s} {'yoy':>7s}")
    for name, fn, fit, active, rel, cv, ym, yp, n_yoy in rows:
        print(f"{name:20s} {fn+format(fit,'+.3f'):>9s} {str(active)+'/32':>7s} "
              f"{rel:>7.3f} {cv:>+7.3f} {ym:>+7.3f}")
    print(f"\nyoy baseline: a pitcher's PAST run prevention predicts next season at "
          f"r={rows[0][7]:+.3f} (n={rows[0][8]} season-pairs).")
    print("A metric is a genuine leading indicator if its yoy column beats that baseline.")


if __name__ == "__main__":
    main()
