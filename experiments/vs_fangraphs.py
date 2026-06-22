"""External validation: our raw-physics whiff-stuff grade vs FanGraphs Stuff+/Pitching+
(2024, pulled via WebFetch into /tmp/fg_2024.csv). Reports Pearson + Spearman.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pybaseball
import torch
import torch.nn as nn
from scipy.stats import spearmanr
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


def main():
    df = load_statcast(FULL_START, dt.date.today().isoformat())
    df = add_fastball_context(mirror_lhp(df))
    df = df[df[INPUT_COLS].notna().all(axis=1)].reset_index(drop=True)
    y = df["description"].isin(WHIFF).to_numpy("float32")
    X = df[INPUT_COLS].to_numpy("float32")
    Xt = torch.from_numpy(StandardScaler().fit_transform(X).astype("float32")).to(DEVICE)
    yt = torch.from_numpy(y).to(DEVICE)

    torch.manual_seed(0)
    model = MLP(X.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    n, bs = len(Xt), 131072
    for _ in range(25):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            opt.zero_grad()
            lossf(model(Xt[idx]), yt[idx]).backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        prob = np.concatenate([torch.sigmoid(model(Xt[i : i + 500_000])).cpu().numpy()
                               for i in range(0, n, 500_000)])
    df["w"] = prob

    d24 = df[df.game_year == 2024]
    g = d24.groupby("pitcher").agg(grade=("w", "mean"), n=("w", "size"),
                                   velo=("release_speed", "mean")).reset_index()
    g = g[g.n >= 300].copy()
    g["grade"] = 100 * g["grade"] / d24["w"].mean()  # index to 100 = league avg

    fg = pd.read_csv("experiments/fg_stuff_2024.csv")
    m = g.merge(fg, left_on="pitcher", right_on="mlbam")
    print(f"merged {len(m)} pitchers (2024)\n")

    for col, lab in [("sp_stuff", "FanGraphs Stuff+"), ("sp_pitching", "FanGraphs Pitching+")]:
        pear = float(np.corrcoef(m["grade"], m[col])[0, 1])
        spear = float(spearmanr(m["grade"], m[col]).correlation)
        print(f"our whiff-stuff vs {lab:20s}:  Pearson {pear:+.3f}   Spearman {spear:+.3f}")
    print(f"(sanity) our grade vs raw velocity      :  Pearson {np.corrcoef(m['grade'], m['velo'])[0,1]:+.3f}")

    lk = pybaseball.playerid_reverse_lookup([int(i) for i in m["pitcher"]], key_type="mlbam")
    lk["name"] = lk["name_first"].str.title() + " " + lk["name_last"].str.title()
    m = m.merge(lk[["key_mlbam", "name"]], left_on="pitcher", right_on="key_mlbam", how="left")
    print("\nside-by-side (sorted by FanGraphs Stuff+):")
    show = m.sort_values("sp_stuff", ascending=False)[["name", "grade", "sp_stuff", "sp_pitching", "velo"]]
    print(show.to_string(index=False, float_format=lambda x: f"{x:.1f}"))


if __name__ == "__main__":
    main()
