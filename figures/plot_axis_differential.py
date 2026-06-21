"""Plot the axis_differential concept column: overall distribution + by pitch type.

Reproduces figures/axis_differential_hist.png. Requires the Statcast cache, so
run the pull first (e.g. `uv run python -m whyplus.model.data --full`). Computing
axis_differential only needs the LHP mirror + the angle formula - no training.

    uv run --with matplotlib python figures/plot_axis_differential.py
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from whyplus.model.data import FULL_START, load_statcast  # noqa: E402
from whyplus.model.features import axis_differential, mirror_lhp  # noqa: E402

OUT = Path(__file__).resolve().parent / "axis_differential_hist.png"


def main() -> None:
    df = load_statcast(FULL_START, dt.date.today().isoformat())  # all cached seasons
    df = mirror_lhp(df)                                          # unified RHP frame
    df = df.assign(axis_differential=axis_differential(df))
    ad = df["axis_differential"].to_numpy()
    ad = ad[np.isfinite(ad)]
    n = len(ad)
    span = f"full {df['game_year'].min()}-{df['game_year'].max()} ({n/1e6:.1f}M pitches)"

    # --- text summary (also useful over SSH) --------------------------------
    pct = np.percentile(ad, [10, 25, 50, 75, 90, 99])
    print(f"axis_differential over {n:,} pitches")
    print("  percentiles 10/25/50/75/90/99 = " + "  ".join(f"{p:.1f}" for p in pct))

    # --- figure -------------------------------------------------------------
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5))

    axA.hist(ad, bins=72, range=(0, 180), color="#3b6fb0", alpha=0.85)
    med_all = float(np.median(ad))
    med_ff = float(np.nanmedian(df.loc[df.pitch_type == "FF", "axis_differential"]))
    axA.axvline(med_all, color="black", ls="--", lw=1.5, label=f"all median {med_all:.1f} deg")
    axA.axvline(med_ff, color="#d1495b", ls="--", lw=1.5, label=f"FF median {med_ff:.1f} deg")
    axA.set_xlabel("axis_differential (deg) = angle between spin-implied & observed movement")
    axA.set_ylabel("pitches")
    axA.set_title(f"(a) All pitches  ({span})")
    axA.legend()

    types = [t for t in ["FF", "SI", "FC", "SL", "ST", "CU", "CH"]
             if (df.pitch_type == t).sum() >= 5000]
    colors = plt.cm.viridis(np.linspace(0, 0.92, len(types)))
    for t, c in zip(types, colors):
        sub = df.loc[df.pitch_type == t, "axis_differential"].to_numpy()
        sub = sub[np.isfinite(sub)]
        axB.hist(sub, bins=60, range=(0, 90), density=True, histtype="step",
                 lw=2, color=c, label=f"{t} (med {np.median(sub):.0f})")
    axB.set_xlabel("axis_differential (deg)")
    axB.set_ylabel("density")
    axB.set_title("(b) By pitch type  (FF tight & low = spin-driven;\n"
                  "breaking/offspeed wider = seam-shifted wake)")
    axB.legend(ncol=2, fontsize=9)

    fig.suptitle("Why+  concept column: axis_differential", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT, dpi=130)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
