"""data.py - pull and cache Statcast pitch data.

Statcast pulls are slow and large (~3.5M rows per full season), so we cache to
parquet and *never re-pull a cached season*. A range that covers a whole season
is cached under ``statcast_{year}.parquet``; a partial range (the smoke window,
or the current in-progress season) is cached under a range-keyed filename so it
can never poison the per-season cache.

We keep only the columns the model and the concept hand-off need, and drop rows
that are null in any raw constant or in the label (``delta_run_exp``). pfx_x /
pfx_z are kept but *not* used as model inputs - they feed concept columns only.

Run standalone to just do the (slow) pull+cache step::

    uv run python -m whyplus.model.data            # 1-week smoke window
    uv run python -m whyplus.model.data --full     # 2021-01-01 -> today
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
from pathlib import Path

import pandas as pd

# --- column contract ---------------------------------------------------------
# Identity columns: who/when/what, plus handedness (drives the LHP mirror).
ID_COLS = ["pitcher", "game_year", "pitch_type", "p_throws"]

# The 12 raw physics constants. These (plus their fb_* means) are the ONLY model
# inputs. No pfx_*, no location, no count, no batter handedness - this is a
# *stuff* model.
RAW_COLS = [
    "release_speed",
    "vx0",
    "vy0",
    "vz0",
    "ax",
    "ay",
    "az",
    "release_pos_x",
    "release_pos_z",
    "release_extension",
    "release_spin_rate",
    "spin_axis",
]

# The label is the negative run-expectancy delta (higher = better for pitcher).
LABEL_COL = "delta_run_exp"

# Movement columns - kept for CONCEPT COLUMNS ONLY. Never model inputs.
CONCEPT_SOURCE_COLS = ["pfx_x", "pfx_z"]

# Statcast publishes arm_angle for recent seasons; we use it where present and
# fall back to a geometric approximation otherwise (handled in features.py).
OPTIONAL_COLS = ["arm_angle"]

# Default smoke window: ~1 week of April 2025 (~50k pitches). Cheap end-to-end
# run + an egress probe for the cloud env.
SMOKE_START = "2025-04-01"
SMOKE_END = "2025-04-08"

# --full lower bound. 2020 is excluded (COVID-short, early-tracking noise).
FULL_START = "2021-01-01"

DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def _statcast():
    """Import pybaseball lazily (heavy import) and enable its on-disk cache."""
    import pybaseball

    pybaseball.cache.enable()
    return pybaseball.statcast


def _keep_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Subset to the column contract; tolerate optional columns being absent."""
    cols = ID_COLS + RAW_COLS + [LABEL_COL] + CONCEPT_SOURCE_COLS
    present_optional = [c for c in OPTIONAL_COLS if c in df.columns]
    cols = cols + present_optional
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"Statcast frame is missing expected columns: {missing}. "
            "pybaseball may have changed its schema."
        )
    return df.loc[:, cols].copy()


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows null in any raw constant, the label, or fields needed to build
    features (pitch_type / p_throws drive the FB context and LHP mirror)."""
    required = RAW_COLS + [LABEL_COL, "pitch_type", "p_throws", "pitcher", "game_year"]
    before = len(df)
    df = df.dropna(subset=required).reset_index(drop=True)
    dropped = before - len(df)
    if before:
        print(f"    dropped {dropped:,}/{before:,} rows null in a raw constant/label "
              f"({dropped / before:.1%})")
    return df


def _season_chunks(start: str, end: str):
    """Split [start, end] into per-year (year, s, e, is_full_season) chunks."""
    s = dt.date.fromisoformat(start)
    e = dt.date.fromisoformat(end)
    if e < s:
        raise ValueError(f"END ({end}) precedes START ({start}).")
    today = dt.date.today()
    for year in range(s.year, e.year + 1):
        jan1 = dt.date(year, 1, 1)
        dec31 = dt.date(year, 12, 31)
        cs = max(s, jan1)
        ce = min(e, dec31)
        # A chunk is a "full season" only if it spans Jan 1 through a date at or
        # past the season's natural end. For the current year the season is
        # in-progress, so treat it as partial (re-pulled, never cached as full).
        season_done = dec31 <= today
        is_full = (cs == jan1) and (ce >= dec31) and season_done
        yield year, cs.isoformat(), ce.isoformat(), is_full


def _cache_path(cache_dir: Path, year: int, s: str, e: str, is_full: bool) -> Path:
    if is_full:
        return cache_dir / f"statcast_{year}.parquet"
    return cache_dir / f"statcast_{year}_{s}_{e}.parquet"


def pull_chunk(year: int, s: str, e: str, is_full: bool, cache_dir: Path) -> pd.DataFrame:
    """Load one per-year chunk from cache, or pull + clean + cache it."""
    path = _cache_path(cache_dir, year, s, e, is_full)
    if path.exists():
        print(f"  [{year}] cache hit  {path.name}")
        return pd.read_parquet(path)

    print(f"  [{year}] pulling {s} -> {e} from Statcast ...")
    statcast = _statcast()
    raw = statcast(start_dt=s, end_dt=e)
    if raw is None or len(raw) == 0:
        print(f"  [{year}] WARNING: empty pull for {s} -> {e}")
        return pd.DataFrame(columns=ID_COLS + RAW_COLS + [LABEL_COL] + CONCEPT_SOURCE_COLS)
    df = _clean(_keep_columns(raw))
    cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"  [{year}] cached {len(df):,} rows -> {path.name}")
    return df


def load_statcast(start: str, end: str, cache_dir: Path = DEFAULT_CACHE_DIR) -> pd.DataFrame:
    """Pull (with per-season caching) and return the concatenated, cleaned frame
    for [start, end]."""
    print(f"Loading Statcast {start} -> {end}  (cache: {cache_dir})")
    frames = [
        pull_chunk(year, s, e, is_full, cache_dir)
        for year, s, e, is_full in _season_chunks(start, end)
    ]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    print(f"Loaded {len(df):,} pitches across {df['game_year'].nunique() if len(df) else 0} season(s).")
    return df


def resolve_range(full: bool, start: str | None, end: str | None) -> tuple[str, str]:
    """Resolve the (START, END) window from flags. Explicit --start/--end win."""
    if start and end:
        return start, end
    if full:
        return FULL_START, dt.date.today().isoformat()
    return SMOKE_START, SMOKE_END


def main() -> None:
    p = argparse.ArgumentParser(description="Pull + cache Statcast pitch data.")
    p.add_argument("--full", action="store_true",
                   help="Pull 2021-01-01 -> today instead of the 1-week smoke window.")
    p.add_argument("--start", default=None, help="Override START (YYYY-MM-DD).")
    p.add_argument("--end", default=None, help="Override END (YYYY-MM-DD).")
    p.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR),
                   help="Where the per-season parquet cache lives.")
    args = p.parse_args()

    start, end = resolve_range(args.full, args.start, args.end)
    df = load_statcast(start, end, Path(args.cache_dir))
    # Tiny summary so a standalone pull is self-verifying.
    if len(df):
        print("\nPer-season row counts:")
        print(df.groupby("game_year").size().to_string())


if __name__ == "__main__":
    main()
