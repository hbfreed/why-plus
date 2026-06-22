"""Side-by-side: our raw-physics xRV grade vs FanGraphs Stuff+ for named 2024 pitchers."""
import datetime as dt

import numpy as np
import pandas as pd
import pybaseball
from sklearn.preprocessing import StandardScaler

from whyplus.model.data import FULL_START, load_statcast
from whyplus.model.features import INPUT_COLS, add_fastball_context, mirror_lhp
from xrv import MLP, build_xrv, pred, train  # reuse the xRV machinery

df = add_fastball_context(mirror_lhp(load_statcast(FULL_START, dt.date.today().isoformat())))
df = df[df[INPUT_COLS].notna().all(axis=1)].reset_index(drop=True)
y = build_xrv(df)
ids = df[["pitcher", "game_year"]].reset_index(drop=True)
X = df[INPUT_COLS].to_numpy("float32")
rng = np.random.default_rng(0)
va = np.zeros(len(df), bool); va[rng.choice(len(df), int(0.15*len(df)), replace=False)] = True
Xs = StandardScaler().fit(X[~va]).transform(X).astype("float32")
m = train(Xs, y, ~va, va, "reg")
grade = pred(m, Xs)

d24 = df["game_year"].to_numpy() == 2024
g = pd.DataFrame({"pitcher": ids["pitcher"][d24], "x": grade[d24]})
g = g.groupby("pitcher")["x"].agg(["mean", "size"]).reset_index()
g = g[g["size"] >= 300].copy()
# put on a Stuff+-like scale: 100 = league avg, ~10 per standard deviation
g["whyplus"] = 100 + 10 * (g["mean"] - g["mean"].mean()) / g["mean"].std()

fg = pd.read_csv("experiments/fg_stuff_2024.csv")
m2 = g.merge(fg, left_on="pitcher", right_on="mlbam")
lk = pybaseball.playerid_reverse_lookup([int(i) for i in m2["pitcher"]], key_type="mlbam")
lk["name"] = lk["name_first"].str.title() + " " + lk["name_last"].str.title()
m2 = m2.merge(lk[["key_mlbam", "name"]], left_on="pitcher", right_on="key_mlbam", how="left")
m2 = m2.rename(columns={"sp_stuff": "FG_Stuff", "sp_pitching": "FG_Pitch"})
m2["diff"] = m2["whyplus"] - m2["FG_Stuff"]

r = np.corrcoef(m2["whyplus"], m2["FG_Stuff"])[0, 1]
print(f"\nr(our xRV grade, FanGraphs Stuff+) = {r:.2f}   (n={len(m2)} pitchers, 2024)\n")
f = lambda x: f"{x:.0f}"
cols = ["name", "whyplus", "FG_Stuff", "FG_Pitch"]
print("TOP 15 by FanGraphs Stuff+ (ours alongside):")
print(m2.sort_values("FG_Stuff", ascending=False).head(15)[cols].to_string(index=False, float_format=f))
print("\nBOTTOM 8:")
print(m2.sort_values("FG_Stuff").head(8)[cols].to_string(index=False, float_format=f))
print("\nWE RATE NOTABLY HIGHER than FG:")
print(m2.sort_values("diff", ascending=False).head(6)[cols].to_string(index=False, float_format=f))
print("\nWE RATE NOTABLY LOWER than FG:")
print(m2.sort_values("diff").head(6)[cols].to_string(index=False, float_format=f))
