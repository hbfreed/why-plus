"""Fun #5b: round-robin feature ablation -> three rankings of a feature's role.

  permutation-importance (from #2): how much the model USES it (with others present)
  leave-one-out (here): how much it HURTS to remove it  = NECESSITY
  single-feature (here): how good it is ALONE            = STANDALONE POWER

A feature can be high-importance but low-necessity (its info is duplicated). The
disagreements between the three rankings reveal redundancy in the physics.

    uv run python experiments/fun_roundrobin.py
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


class MLP(nn.Module):
    def __init__(self, d_in, d_repr=32, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, d_repr), nn.ReLU(), nn.Linear(d_repr, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


df = add_fastball_context(mirror_lhp(load_statcast(FULL_START, dt.date.today().isoformat())))
df = df[df[INPUT_COLS].notna().all(axis=1)].reset_index(drop=True)
y = build_xrv(df)
rng = np.random.default_rng(0)
va = np.zeros(len(df), bool); va[rng.choice(len(df), int(0.15 * len(df)), replace=False)] = True
tr = ~va
yv = y[va]


def r2(cols):
    X = df[cols].to_numpy("float32")
    sc = StandardScaler().fit(X[tr]); Xs = sc.transform(X).astype("float32")
    torch.manual_seed(0)
    m = MLP(len(cols)).to(DEVICE)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-4)
    lossf = nn.HuberLoss(delta=1.0)
    Xtr = torch.from_numpy(Xs[tr]).to(DEVICE); ytr = torch.from_numpy(y[tr]).to(DEVICE)
    Xva = torch.from_numpy(Xs[va]).to(DEVICE); yva = torch.from_numpy(yv).to(DEVICE)
    n, best, state, bad = len(Xtr), 1e9, None, 0
    for _ in range(22):
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
            if bad >= 4:
                break
    m.load_state_dict(state); m.eval()
    with torch.no_grad():
        pv = m(Xva).cpu().numpy()
    return float(1 - ((yv - pv) ** 2).sum() / ((yv - yv.mean()) ** 2).sum())


base = r2(INPUT_COLS)
print(f"baseline R2 (all 24) = {base:.4f}\n")

loo = []   # necessity: R2 drop when removed
solo = []  # standalone: R2 alone
for c in INPUT_COLS:
    loo.append((c, base - r2([x for x in INPUT_COLS if x != c])))
    solo.append((c, r2([c])))

print("=== NECESSITY (leave-one-out R2 drop; bigger = harder to replace) ===")
for c, d in sorted(loo, key=lambda t: -t[1]):
    print(f"  {c:24s} {d:+.4f}")
print("\n=== STANDALONE power (R2 with that feature ALONE) ===")
for c, v in sorted(solo, key=lambda t: -t[1]):
    print(f"  {c:24s} {v:.4f}")
