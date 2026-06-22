"""Face validity + external check for the whiff-stuff grade. No smoke.

Train a whiff model on raw physics, grade every pitcher-season, NAME the leaders,
sanity-check the bottom, test whether it's just a radar gun (corr with velocity),
and -- if FanGraphs is reachable -- correlate against their Stuff+.

    uv run python experiments/face_validity.py
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pybaseball
import torch
import torch.nn as nn
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
            nn.Linear(128, d_repr), nn.ReLU(),
            nn.Linear(d_repr, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def main():
    print(f"Device: {DEVICE}")
    df = load_statcast(FULL_START, dt.date.today().isoformat())
    df = add_fastball_context(mirror_lhp(df))
    df = df[df[INPUT_COLS].notna().all(axis=1)].reset_index(drop=True)
    y = df["description"].isin(WHIFF).to_numpy("float32")
    X = df[INPUT_COLS].to_numpy("float32")
    sc = StandardScaler().fit(X)
    Xt = torch.from_numpy(sc.transform(X).astype("float32")).to(DEVICE)
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
    df["wstuff"] = prob

    g = df.groupby(["pitcher", "game_year"]).agg(
        wstuff=("wstuff", "mean"), velo=("release_speed", "mean"), n=("wstuff", "size")
    ).reset_index()
    g = g[g.n >= 500].copy()
    g["grade"] = 100 * g["wstuff"] / df["wstuff"].mean()

    # honesty check: is it just a radar gun?
    r_velo = float(np.corrcoef(g["grade"], g["velo"])[0, 1])
    print(f"\n{len(g)} pitcher-seasons graded (>=500 pitches).")
    print(f"corr(grade, avg velocity) = {r_velo:.3f}   "
          "(near 1.0 => it's mostly a radar gun; lower => it found more than velo)")

    # names
    ids = [int(i) for i in g["pitcher"].unique()]
    lk = pybaseball.playerid_reverse_lookup(ids, key_type="mlbam")
    lk["name"] = lk["name_first"].str.title() + " " + lk["name_last"].str.title()
    g = g.merge(lk[["key_mlbam", "name", "key_fangraphs"]],
                left_on="pitcher", right_on="key_mlbam", how="left")

    cols = ["name", "game_year", "n", "grade", "velo"]
    fmt = lambda x: f"{x:.1f}"
    print("\n=== TOP 25 whiff-stuff (raw physics only) ===")
    print(g.sort_values("grade", ascending=False).head(25)[cols].to_string(index=False, float_format=fmt))
    print("\n=== BOTTOM 10 (sanity: expect soft-tossers / position players) ===")
    print(g.sort_values("grade").head(10)[cols].to_string(index=False, float_format=fmt))

    # external check: FanGraphs Stuff+
    try:
        fg = []
        for yr in range(2021, 2026):
            s = pybaseball.pitching_stats(yr, qual=20)
            if "Stuff+" in s.columns:
                fg.append(s[["IDfg", "Season", "Stuff+"]])
        if not fg:
            print("\nFanGraphs returned no 'Stuff+' column; external check skipped.")
            return
        fg = pd.concat(fg)
        fg["IDfg"] = pd.to_numeric(fg["IDfg"], errors="coerce")
        g["key_fangraphs"] = pd.to_numeric(g["key_fangraphs"], errors="coerce")
        cmp = g.merge(fg, left_on=["key_fangraphs", "game_year"], right_on=["IDfg", "Season"])
        cmp = cmp.dropna(subset=["grade", "Stuff+"])
        r = float(np.corrcoef(cmp["grade"], cmp["Stuff+"])[0, 1])
        print(f"\n=== vs FanGraphs Stuff+  (n={len(cmp)} pitcher-seasons)  r = {r:.3f} ===")
        print("  (r~0.6-0.8 = strong agreement; whiff-stuff != full Stuff+, so not 1.0)")
        show = cmp.sort_values("grade", ascending=False).head(15)[["name", "game_year", "grade", "Stuff+"]]
        print(show.to_string(index=False, float_format=fmt))
    except Exception as e:
        print(f"\nFanGraphs Stuff+ check unavailable: {type(e).__name__}: {str(e)[:140]}")


if __name__ == "__main__":
    main()
