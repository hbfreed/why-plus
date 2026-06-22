# experiments/ — exploration beyond the brief

The brief's Stuff+ (run-value target, physics-only inputs) lives in `whyplus/model/`.
These scripts explore whether a **better, more explainable** pitch stat is reachable
from the same raw physics — and validate it against the industry metric.

Run from the repo root: `uv run python experiments/<script>.py`

## The headline (2026-06-21)

**Judge a pitch stat by reliability + predictiveness, not per-pitch accuracy.** Per
pitch is ~irreducible noise (ceiling ~0.75 AUC / ~0.04 R²); the signal lives in the
season-level *average* over a pitcher's pitches.

| script | what it showed |
|---|---|
| `bakeoff.py` | 2×2 inputs×target. Architecture is noise; **target + inputs** are the only levers. |
| `sweep.py` | 72 configs (AdamW). `d_repr`/dropout/`weight_decay` all within ±0.005 yoy. `whiff-per-swing + pitch+fb` won the whiff family. |
| `ceiling_gbm.py` | A gradient-boosted tree can't beat the MLP (0.73 vs 0.76 AUC) → we're at the **data ceiling, not underfit**. |
| `face_validity.py` | Leaderboard passes: deGrom #1, Devin Williams elite at 88 mph, submariners at the floor. velo corr **0.40** (not a radar gun). |
| `vs_fangraphs.py` | whiff grade vs FanGraphs Stuff+: **r = 0.63**. |
| `xrv.py` | **Winner.** xRV target (run value with batted balls replaced by *expected* RV from xwOBA) **doubles R² (0.02→0.04)**, reliability 0.82, and **r = 0.84 with FanGraphs Stuff+**. |
| `compare_players.py` | Player side-by-side. **Clase: 111 (whiff) → 132 (xRV) ≈ FG 128** — xRV captures the weak-contact stuff whiff is blind to. |
| `save_xrv_model.py` | Trains the chosen model and saves `artifacts/xrv_model.pt`. |

**Chosen model:** xRV target, `pitch+fb` inputs, `d_repr=32`, AdamW.

## Notes / honesty
- Inputs are **always raw physics only** — no constructed/derived features (no diffs,
  movement, tunnels). xRV touches only the *label*, never an input.
- FanGraphs blocks this box (Cloudflare); Stuff+ was pulled via WebFetch into
  `fg_stuff_2024.csv` (n=66, 2024). Lossy + small — treat individual values as ±a few;
  the **r = 0.84 and the rankings** are the solid part.
- Grades are in-sample (fine for ranking/face-validity); reliability/yoy are the
  out-of-sample numbers.
