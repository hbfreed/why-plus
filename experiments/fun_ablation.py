"""Fun #5: take out the features the model leans on. Importance != necessity?

The inputs are highly correlated (velocity lives in vx0/vy0/vz0/release_speed; movement
in ax/az), so removing a top feature may barely hurt -- the model recovers it from the
survivors. We retrain with the top-k most-important features removed, and contrast with
removing the bottom-k, to see when it actually collapses.

    uv run python experiments/fun_ablation.py
"""
import datetime as dt

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from whyplus.model.data import FULL_START, load_statcast
from whyplus.model.features import INPUT_COLS, add_fastball_context, mirror_lhp
from xrv import build_xrv

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# importance order (top -> bottom) from fun_pdp.py
IMP_ORDER = ["vx0", "vz0", "release_pos_x", "ax", "az", "release_pos_z", "vy0",
             "release_speed", "fb_release_speed", "spin_axis", "fb_release_pos_z",
             "fb_vx0", "fb_vz0", "fb_spin_axis", "fb_vy0", "fb_az", "fb_release_pos_x",
             "ay", "release_spin_rate", "release_extension", "fb_ay", "fb_ax",
             "fb_release_extension", "fb_release_spin_rate"]


class MLP(nn.Module):
    def __init__(self, d_in, d_repr=32, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, d_repr), nn.ReLU(), nn.Linear(d_repr, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def reliability(ids, p, min_n=100):
    rng = np.random.default_rng(0)
    d = ids.copy(); d["p"] = p; d["h"] = rng.integers(0, 2, len(p))
    cnt = d.groupby(["pitcher", "game_year"])["p"].transform("size"); d = d[cnt >= min_n]
    piv = d.groupby(["pitcher", "game_year", "h"])["p"].mean().unstack("h").dropna()
    return float(np.corrcoef(piv[0], piv[1])[0, 1]) if len(piv) >= 10 else float("nan")


df = add_fastball_context(mirror_lhp(load_statcast(FULL_START, dt.date.today().isoformat())))
df = df[df[INPUT_COLS].notna().all(axis=1)].reset_index(drop=True)
y = build_xrv(df)
ids = df[["pitcher", "game_year"]].reset_index(drop=True)
rng = np.random.default_rng(0)
va = np.zeros(len(df), bool); va[rng.choice(len(df), int(0.15 * len(df)), replace=False)] = True
tr = ~va


def train_eval(cols):
    X = df[cols].to_numpy("float32")
    sc = StandardScaler().fit(X[tr])
    Xs = sc.transform(X).astype("float32")
    torch.manual_seed(0)
    m = MLP(len(cols)).to(DEVICE)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-4)
    lossf = nn.HuberLoss(delta=1.0)
    Xtr = torch.from_numpy(Xs[tr]).to(DEVICE); ytr = torch.from_numpy(y[tr]).to(DEVICE)
    Xva = torch.from_numpy(Xs[va]).to(DEVICE); yva = torch.from_numpy(y[va]).to(DEVICE)
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
        pv = m(Xva).cpu().numpy()
        pa = np.concatenate([m(torch.from_numpy(Xs[i:i+500000]).to(DEVICE)).cpu().numpy()
                             for i in range(0, len(Xs), 500000)])
    yv = y[va]
    r2 = 1 - ((yv - pv) ** 2).sum() / ((yv - yv.mean()) ** 2).sum()
    return float(r2), reliability(ids, pa)


base_r2, base_rel = train_eval(INPUT_COLS)
print(f"baseline (all 24 features):   R2={base_r2:.4f}  reliab={base_rel:.3f}\n")
print(f"{'k':>3s} | remove TOP-k (the ones it leans on) | remove BOTTOM-k (least used)")
print(f"{'':>3s} |    R2      reliab   kept             |    R2      reliab")
for k in [2, 4, 6, 8, 10, 12]:
    top_kept = [c for c in INPUT_COLS if c not in IMP_ORDER[:k]]
    bot_kept = [c for c in INPUT_COLS if c not in IMP_ORDER[-k:]]
    rt, lt = train_eval(top_kept)
    rb, lb = train_eval(bot_kept)
    print(f"{k:>3d} | {rt:.4f}   {lt:.3f}   ({len(top_kept)} left)      | {rb:.4f}   {lb:.3f}")

# extreme: blind to ALL velocity + movement (keep only release point, spin, extension, axis, fb)
blind = [c for c in INPUT_COLS if c not in
         {"vx0", "vy0", "vz0", "release_speed", "ax", "ay", "az"}]
rbl, lbl = train_eval(blind)
print(f"\nBLIND to velocity+movement (keep release pos/spin/ext/axis + fb, {len(blind)} feats):"
      f"  R2={rbl:.4f}  reliab={lbl:.3f}")
