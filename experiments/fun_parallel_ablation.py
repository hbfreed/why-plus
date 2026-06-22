"""Fun #5c: a BAZILLION in parallel.

Train EVERY ablation model at once in a single vmapped pass: baseline + all 24
leave-one-out + all 24 single-feature = 49 tiny nets trained simultaneously on one
GPU (the nets are so small that sequential training wastes the hardware).

Ablation = mean-masking on standardized inputs (zero a column = hold it at its mean),
so every model keeps d_in=24 and they can be stacked and vmapped together.

Three feature rankings fall out:
  permutation-importance (#2) : how much the model USES a feature
  leave-one-out necessity     : how much removing it HURTS (low => redundant)
  standalone power            : how good the feature is ALONE

    uv run python experiments/fun_parallel_ablation.py
"""
import copy
import datetime as dt

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.func import functional_call, stack_module_state, vmap

from whyplus.model.data import FULL_START, load_statcast
from whyplus.model.features import INPUT_COLS, add_fastball_context, mirror_lhp
from xrv import build_xrv

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class MLP(nn.Module):  # no dropout -> deterministic, clean under vmap
    def __init__(self, d_in=24, d_repr=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, d_repr), nn.ReLU(), nn.Linear(d_repr, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


df = add_fastball_context(mirror_lhp(load_statcast(FULL_START, dt.date.today().isoformat())))
df = df[df[INPUT_COLS].notna().all(axis=1)].reset_index(drop=True)
y = build_xrv(df)
X = df[INPUT_COLS].to_numpy("float32")
rng = np.random.default_rng(0)
va = np.zeros(len(df), bool); va[rng.choice(len(df), int(0.15 * len(df)), replace=False)] = True
tr = ~va
Xs = StandardScaler().fit(X[tr]).transform(X).astype("float32")
Xtr = torch.from_numpy(Xs[tr]).to(DEVICE); ytr = torch.from_numpy(y[tr]).to(DEVICE)
Xva = torch.from_numpy(Xs[va]).to(DEVICE); yva = torch.from_numpy(y[va]).to(DEVICE)
nf = len(INPUT_COLS)

# build the ablation masks
labels, masks = ["baseline"], [np.ones(nf, "float32")]
for i, c in enumerate(INPUT_COLS):
    mm = np.ones(nf, "float32"); mm[i] = 0; labels.append("LOO:" + c); masks.append(mm)
for i, c in enumerate(INPUT_COLS):
    mm = np.zeros(nf, "float32"); mm[i] = 1; labels.append("solo:" + c); masks.append(mm)
M = torch.from_numpy(np.stack(masks)).to(DEVICE)  # (N, nf)
N = len(masks)
print(f"training {N} models in parallel via vmap on {DEVICE}\n")

torch.manual_seed(0)
models = [MLP().to(DEVICE) for _ in range(N)]
params, _ = stack_module_state(models)
params = {k: v.detach().requires_grad_(True) for k, v in params.items()}
base = copy.deepcopy(models[0]).to("meta")


def fwd(p, mask, x):
    return functional_call(base, (p,), (x * mask,))


vfwd = vmap(fwd, in_dims=(0, 0, None))
opt = torch.optim.AdamW(list(params.values()), lr=2e-3, weight_decay=1e-4)
huber = nn.HuberLoss(delta=1.0)

ntr, bs = len(Xtr), 16384
for ep in range(22):
    perm = torch.randperm(ntr, device=DEVICE)
    for i in range(0, ntr, bs):
        idx = perm[i:i+bs]
        opt.zero_grad()
        pred = vfwd(params, M, Xtr[idx])              # (N, b)
        huber(pred, ytr[idx].unsqueeze(0).expand(N, -1)).backward()
        opt.step()

with torch.no_grad():
    pv = torch.cat([vfwd(params, M, Xva[i:i+100000]) for i in range(0, len(Xva), 100000)], dim=1)
pv = pv.cpu().numpy()
yv = yva.cpu().numpy()
sst = ((yv - yv.mean()) ** 2).sum()
r2 = 1 - ((yv[None, :] - pv) ** 2).sum(1) / sst
res = dict(zip(labels, r2))
b = res["baseline"]

print(f"baseline R2 (all 24 features) = {b:.4f}\n")
print("=== NECESSITY: leave-one-out R2 drop (big = hard to replace; small = redundant) ===")
for c, d in sorted(((c, b - res["LOO:" + c]) for c in INPUT_COLS), key=lambda t: -t[1]):
    print(f"  {c:24s} {d:+.4f}")
print("\n=== STANDALONE: R2 with ONLY that feature ===")
for c, v in sorted(((c, res["solo:" + c]) for c in INPUT_COLS), key=lambda t: -t[1]):
    print(f"  {c:24s} {v:.4f}")
