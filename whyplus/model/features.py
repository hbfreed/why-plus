"""features.py - turn cleaned Statcast rows into model inputs + concept columns.

ORDER MATTERS (per the brief):
  1. Mirror LHP into a unified RHP frame FIRST, so fastball means live in one frame.
  2. Fastball context: per (pitcher, game_year), pick the primary fastball and
     broadcast its raw-constant means to every pitch as fb_* columns. The net is
     meant to learn velo/movement/tunnel *differentials* itself - we do NOT feed
     precomputed diffs.
  3. Inputs X = raw constants + fb_* means (~24 dims). NO pfx_*, location, count,
     or batter handedness - this is a *stuff* model.
  4. Label y = -delta_run_exp (higher = better for the pitcher).
  5. Concept columns - computed here ONLY so they align row-for-row with the
     representations for the NLA brief. Never inputs, never probed here.

The product of this module is four row-aligned frames: X, y, concepts, ids.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import LABEL_COL, RAW_COLS

# Pitch types that count as a "true" fastball for the FB-context reference.
FB_TYPES = {"FF", "SI", "FC"}

# Columns negated when reflecting a LHP into the RHP frame. These are the
# horizontal/lateral physics terms; vertical and speed terms are handedness-
# symmetric and left alone.
MIRROR_NEGATE = ["vx0", "ax", "release_pos_x"]

# Final model-input columns: the raw constants plus their fastball-context means.
FB_COLS = [f"fb_{c}" for c in RAW_COLS]
INPUT_COLS = RAW_COLS + FB_COLS

# The concept columns handed to the NLA brief (aligned, never inputs).
CONCEPT_COLS = ["pfx_x", "pfx_z", "vaa", "release_extension", "arm_angle", "axis_differential"]

# Geometry of the measurement frame.
Y_RELEASE_PLANE = 50.0   # ft; Statcast reports vy0/etc. at y=50
Y_PLATE_FRONT = 17.0 / 12.0  # ft; front edge of home plate

# Rough shoulder pivot for the arm-angle *fallback* approximation (RHP frame).
# Only used where Statcast's own arm_angle is absent (older seasons).
SHOULDER_Z_APPROX = 5.2  # ft
SHOULDER_X_APPROX = 1.0  # ft toward the throwing-arm side of body center


# --- 1. LHP mirror -----------------------------------------------------------
def mirror_lhp(df: pd.DataFrame) -> pd.DataFrame:
    """Reflect left-handed pitchers into a unified RHP frame, IN PLACE on a copy.

    Negates the lateral physics terms and flips spin_axis across the vertical.
    Also flips pfx_x (a concept source) so the concept frame stays consistent
    with the mirrored spin_axis - otherwise axis_differential is wrong for LHP.
    """
    df = df.copy()
    is_lhp = df["p_throws"].eq("L")
    n = int(is_lhp.sum())
    if n:
        for col in MIRROR_NEGATE:
            df.loc[is_lhp, col] = -df.loc[is_lhp, col]
        # spin_axis is a compass bearing in [0,360); reflect across the vertical.
        df.loc[is_lhp, "spin_axis"] = (360.0 - df.loc[is_lhp, "spin_axis"]) % 360.0
        # pfx_x is a concept source, not a model input, but must share the frame.
        if "pfx_x" in df.columns:
            df.loc[is_lhp, "pfx_x"] = -df.loc[is_lhp, "pfx_x"]
    print(f"  mirrored {n:,} LHP pitches into the RHP frame")
    return df


# --- 2. Fastball context -----------------------------------------------------
def add_fastball_context(df: pd.DataFrame) -> pd.DataFrame:
    """Broadcast each pitcher-season's primary-fastball raw-constant means to
    every pitch as fb_* columns.

    Primary FB = highest-usage of {FF, SI, FC}. Edge case: a pitcher-season with
    no FF/SI/FC uses its highest-usage pitch as the reference, and those rows are
    flagged in ``fb_is_fallback`` (HUMAN DECISION: reference, not drop).
    """
    keys = ["pitcher", "game_year"]

    # Usage counts per (pitcher, season, pitch_type).
    usage = df.groupby(keys + ["pitch_type"], observed=True).size().rename("n").reset_index()
    usage["is_fb"] = usage["pitch_type"].isin(FB_TYPES)

    # Within each pitcher-season, prefer a fastball, then highest usage. A stable
    # sort by (is_fb desc, n desc) means .first() per group is exactly the primary.
    usage = usage.sort_values(
        keys + ["is_fb", "n"], ascending=[True, True, False, False]
    )
    primary = usage.groupby(keys, observed=True, sort=False).first().reset_index()
    primary = primary.rename(columns={"pitch_type": "fb_type"})
    primary["fb_is_fallback"] = ~primary["is_fb"]
    n_fallback = int(primary["fb_is_fallback"].sum())
    print(f"  {len(primary):,} pitcher-seasons; "
          f"{n_fallback:,} have no FF/SI/FC -> highest-usage fallback reference")

    # Mean of each raw constant per (pitcher, season, pitch_type) ...
    means = (
        df.groupby(keys + ["pitch_type"], observed=True)[RAW_COLS].mean().reset_index()
    )
    # ... selected for the primary pitch type of each pitcher-season.
    fb_means = means.merge(
        primary[keys + ["fb_type"]],
        left_on=keys + ["pitch_type"],
        right_on=keys + ["fb_type"],
        how="inner",
    )
    fb_means = fb_means[keys + RAW_COLS].rename(columns={c: f"fb_{c}" for c in RAW_COLS})

    out = df.merge(fb_means, on=keys, how="left")
    out = out.merge(primary[keys + ["fb_is_fallback"]], on=keys, how="left")
    return out


# --- 5. Concept columns (aligned, never inputs) ------------------------------
def vertical_approach_angle(df: pd.DataFrame) -> np.ndarray:
    """VAA at the front of home plate, in degrees (negative = downward).

    Propagates the 9-parameter trajectory from the y=50 ft measurement plane to
    the plate using v_f^2 = v_i^2 + 2*a*dy, then takes the velocity-vector angle
    below horizontal. Derived from raw constants only - a concept, never an input.
    """
    vy0, vz0 = df["vy0"].to_numpy(), df["vz0"].to_numpy()
    ay, az = df["ay"].to_numpy(), df["az"].to_numpy()
    # Ball travels toward home (decreasing y), so vy_f is negative.
    disc = vy0**2 - 2.0 * ay * (Y_RELEASE_PLANE - Y_PLATE_FRONT)
    vy_f = -np.sqrt(np.clip(disc, 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (vy_f - vy0) / ay
        vz_f = vz0 + az * t
        # angle below horizontal: horizontal speed is |vy_f|.
        vaa = np.degrees(np.arctan2(vz_f, np.abs(vy_f)))
    return vaa


def arm_angle(df: pd.DataFrame) -> np.ndarray:
    """Arm angle in degrees (0 = sidearm, 90 = over-the-top, negative = submarine).

    Prefer Statcast's own ``arm_angle`` where present; otherwise approximate from
    the (mirrored) release point relative to a rough shoulder pivot. The fallback
    is a coarse geometric proxy - fine for a concept column the NLA brief probes.
    """
    n = len(df)
    rel_x = df["release_pos_x"].to_numpy()  # already mirrored to RHP frame
    rel_z = df["release_pos_z"].to_numpy()
    approx = np.degrees(
        np.arctan2(rel_z - SHOULDER_Z_APPROX, np.abs(rel_x - SHOULDER_X_APPROX))
    )
    if "arm_angle" in df.columns:
        statcast = df["arm_angle"].to_numpy(dtype=float)
        out = np.where(np.isfinite(statcast), statcast, approx)
        n_fallback = int((~np.isfinite(statcast)).sum())
        if n_fallback:
            print(f"  arm_angle: {n_fallback:,}/{n:,} rows use the geometric fallback")
        else:
            print("  arm_angle: using Statcast's column for all rows")
        return out
    print(f"  arm_angle: Statcast column absent; geometric fallback for all {n:,} rows")
    return approx


def axis_differential(df: pd.DataFrame) -> np.ndarray:
    """Angle (deg, [0,180]) between spin-IMPLIED movement and OBSERVED movement.

    A seam-shifted-wake proxy. We compare the movement direction implied by the
    spin axis to the observed pfx movement direction (removing the inherent 90
    deg offset between an axis and the force perpendicular to it).

    Convention is pinned on the unambiguous vertical axis: pure backspin
    (spin_axis=180) implies movement straight up (+pfx_z); pure topspin
    (spin_axis=0/360) straight down. Both poles fix only the vertical; the
    horizontal handedness is set by the data: raw Statcast pfx_x is *negative*
    on the arm side (a RHP four-seamer averages pfx_x ~= -0.6 at spin_axis ~214),
    which gives spin_mov_angle = spin_axis - 90 in the atan2(pfx_z, pfx_x) frame.
    Verified empirically: this collapses the FF median differential to ~10 deg
    (vs ~58 deg for the wrong sign). Both spin_axis and pfx_x are already in the
    mirrored RHP frame, and an angular magnitude is reflection-invariant, so this
    is consistent across handedness.
    """
    if not {"pfx_x", "pfx_z"}.issubset(df.columns):
        return np.full(len(df), np.nan)
    pfx_x, pfx_z = df["pfx_x"].to_numpy(), df["pfx_z"].to_numpy()
    spin_axis = df["spin_axis"].to_numpy()
    obs = np.degrees(np.arctan2(pfx_z, pfx_x))
    implied = (spin_axis - 90.0) % 360.0
    diff = np.abs(((obs - implied + 180.0) % 360.0) - 180.0)
    return diff


def build_concepts(df: pd.DataFrame) -> pd.DataFrame:
    """Assemble the row-aligned concept frame. Computed here only; probed later."""
    concepts = pd.DataFrame(index=df.index)
    concepts["pfx_x"] = df["pfx_x"] if "pfx_x" in df.columns else np.nan  # mirrored frame
    concepts["pfx_z"] = df["pfx_z"] if "pfx_z" in df.columns else np.nan
    concepts["vaa"] = vertical_approach_angle(df)
    concepts["release_extension"] = df["release_extension"]
    concepts["arm_angle"] = arm_angle(df)
    concepts["axis_differential"] = axis_differential(df)

    # Quick convention check: four-seamers are spin-dominated, so their
    # axis_differential should be SMALL. A large median here means the spin->
    # movement convention is flipped.
    ff = df["pitch_type"].eq("FF")
    if ff.any():
        med_ff = float(np.nanmedian(concepts.loc[ff.to_numpy(), "axis_differential"]))
        med_all = float(np.nanmedian(concepts["axis_differential"]))
        print(f"  axis_differential sanity: FF median={med_ff:.1f} deg, "
              f"all median={med_all:.1f} deg (FF should be the smaller / modest)")
    return concepts


# --- top-level assembly ------------------------------------------------------
def build_features(df: pd.DataFrame):
    """Run the full ordered pipeline.

    Returns
    -------
    X : DataFrame  (n, ~24) model inputs (raw constants + fb_* means)
    y : Series     (n,)     label = -delta_run_exp
    concepts : DataFrame    row-aligned concept columns (+ fb_is_fallback flag)
    ids : DataFrame         pitcher / game_year / pitch_type / p_throws
    """
    print("Building features ...")
    df = mirror_lhp(df)              # step 1
    df = add_fastball_context(df)    # step 2

    # A pitcher-season with a single pitch of the primary type still yields fb_*
    # means (its own values); fb_* should never be null after the left-join, but
    # guard anyway so a silent NaN can't leak into the model inputs.
    n_bad = int(df[FB_COLS].isna().any(axis=1).sum())
    if n_bad:
        print(f"  WARNING: dropping {n_bad:,} rows with null fb_* context")
        df = df[~df[FB_COLS].isna().any(axis=1)].reset_index(drop=True)

    X = df[INPUT_COLS].copy()                 # step 3
    y = (-df[LABEL_COL]).rename("run_value")  # step 4
    concepts = build_concepts(df)             # step 5
    concepts["fb_is_fallback"] = df["fb_is_fallback"].to_numpy()
    concepts["delta_run_exp"] = df[LABEL_COL].to_numpy()
    concepts["run_value"] = y.to_numpy()
    ids = df[["pitcher", "game_year", "pitch_type", "p_throws"]].copy()

    print(f"  X: {X.shape}  (inputs={len(INPUT_COLS)})  |  concepts: {concepts.shape}")
    assert len(X) == len(y) == len(concepts) == len(ids), "row alignment broken"
    return X, y, concepts, ids
