"""train.py - train the StuffMLP and write the NLA hand-off artifact.

This is the whole exit criterion for the model brief: train a from-scratch
Stuff+, save a clean hand-off, and run a *pure-model* non-degeneracy check. We do
NOT optimize for predictive accuracy - low per-pitch val R^2 (~0.05-0.15) is the
noise floor, not failure. Whether the representation is worth explaining is the
NLA brief's first question, not this one.

    uv run python -m whyplus.model.train              # 1-week smoke run
    uv run python -m whyplus.model.train --full       # 2021 -> today
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from .data import DEFAULT_CACHE_DIR, load_statcast, resolve_range
from .features import CONCEPT_COLS, INPUT_COLS, build_features
from .net import StuffMLP

DEFAULT_OUT_DIR = Path(__file__).resolve().parents[2] / "artifacts"
TEMPORAL_VAL_YEAR = 2025  # temporal split: train < this year, val >= it


# --- splitting ---------------------------------------------------------------
def make_split(ids: pd.DataFrame, how: str, val_frac: float, seed: int) -> np.ndarray:
    """Return a boolean mask that is True for validation rows.

    Random (the v0 default) or temporal (train <=2024 / val 2025). Temporal falls
    back to random if it would empty either side (e.g. the single-season smoke
    window), since a degenerate split can't validate anything.
    """
    n = len(ids)
    rng = np.random.default_rng(seed)
    if how == "temporal":
        is_val = (ids["game_year"].to_numpy() >= TEMPORAL_VAL_YEAR)
        if is_val.all() or not is_val.any():
            print(f"  temporal split degenerate (all rows on one side of "
                  f"{TEMPORAL_VAL_YEAR}); falling back to random {val_frac:.0%}")
        else:
            print(f"  temporal split: train < {TEMPORAL_VAL_YEAR}, val >= {TEMPORAL_VAL_YEAR} "
                  f"({is_val.mean():.1%} val)")
            return is_val
    # random
    val_mask = np.zeros(n, dtype=bool)
    val_idx = rng.choice(n, size=int(round(val_frac * n)), replace=False)
    val_mask[val_idx] = True
    print(f"  random split: {val_mask.mean():.1%} held out for validation")
    return val_mask


# --- training ----------------------------------------------------------------
def train_model(
    Xtr, ytr, Xva, yva, *, d_repr, epochs, batch_size, lr, weight_decay,
    huber_delta, patience, device, seed,
):
    """Train with Huber loss + Adam, early-stopping on val. Returns the model
    (best weights restored) and a small history dict."""
    torch.manual_seed(seed)
    model = StuffMLP(d_in=Xtr.shape[1], d_repr=d_repr).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # Huber / smooth-L1: delta_run_exp is heavy-tailed from terminal events, so
    # MSE would overweight rare big-RV pitches. Huber is linear past |err|>delta.
    loss_fn = nn.HuberLoss(delta=huber_delta)

    tr = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
    loader = DataLoader(tr, batch_size=batch_size, shuffle=True, drop_last=False)
    Xva_t = torch.from_numpy(Xva).to(device)
    yva_t = torch.from_numpy(yva).to(device)

    best_val = float("inf")
    best_state = None
    bad_epochs = 0
    history = {"train": [], "val": []}

    for epoch in range(1, epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred, _ = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            running += loss.item() * len(xb)
            seen += len(xb)
        train_loss = running / max(seen, 1)

        model.eval()
        with torch.no_grad():
            vpred, _ = model(Xva_t)
            val_loss = loss_fn(vpred, yva_t).item()
        history["train"].append(train_loss)
        history["val"].append(val_loss)

        improved = val_loss < best_val - 1e-6
        if improved:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        flag = "  *" if improved else ""
        print(f"  epoch {epoch:3d}  train {train_loss:.5f}  val {val_loss:.5f}{flag}")
        if bad_epochs >= patience:
            print(f"  early stop (no val improvement in {patience} epochs)")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best_val


# --- representation extraction ----------------------------------------------
@torch.no_grad()
def extract_representations(model, X, device, batch_size=16384) -> np.ndarray:
    """Penultimate (d_repr) vectors for every row, in input order."""
    model.eval()
    out = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i : i + batch_size]).to(device)
        out.append(model.represent(xb).cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


# --- sanity check (the whole exit criterion) --------------------------------
def sanity_check(model, Xva, yva, ytr, reps, huber_delta, device) -> dict:
    """Pure-model non-degeneracy check: (1) val beats a constant baseline, and
    (2) the penultimate representation has non-trivial variance (not collapsed).
    No concepts, no probing - that is the NLA brief."""
    loss_fn = nn.HuberLoss(delta=huber_delta)

    model.eval()
    with torch.no_grad():
        vpred, _ = model(torch.from_numpy(Xva).to(device))
        vpred = vpred.cpu().numpy()
    yva_t = torch.from_numpy(yva)
    model_loss = loss_fn(torch.from_numpy(vpred), yva_t).item()

    # Constant baseline: predict mean(y_train) for everything.
    const = np.full_like(yva, fill_value=float(ytr.mean()))
    base_loss = loss_fn(torch.from_numpy(const), yva_t).item()

    # R^2 for context only (NOT a pass/fail; expected low ~0.05-0.15).
    ss_res = float(np.sum((yva - vpred) ** 2))
    ss_tot = float(np.sum((yva - yva.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # Representation health.
    per_dim_std = reps.std(axis=0)
    active = int((per_dim_std > 1e-4).sum())
    beats_baseline = model_loss < base_loss
    not_collapsed = active >= 2 and float(per_dim_std.max()) > 1e-3

    result = {
        "val_huber_model": model_loss,
        "val_huber_baseline": base_loss,
        "beats_baseline": bool(beats_baseline),
        "val_r2": r2,
        "repr_active_dims": active,
        "repr_total_dims": int(reps.shape[1]),
        "repr_mean_std": float(per_dim_std.mean()),
        "repr_max_std": float(per_dim_std.max()),
        "not_collapsed": bool(not_collapsed),
        "passed": bool(beats_baseline and not_collapsed),
    }

    print("\n=== SANITY CHECK (pure-model; the exit criterion) ===")
    print(f"  val Huber  : model {model_loss:.5f}  vs  constant-baseline {base_loss:.5f}"
          f"  -> {'BEATS' if beats_baseline else 'FAILS'} baseline")
    print(f"  val R^2    : {r2:.4f}   (context only - low is the noise floor, not failure)")
    print(f"  repr health: {active}/{reps.shape[1]} active dims, "
          f"mean std {per_dim_std.mean():.4f}, max std {per_dim_std.max():.4f}"
          f"  -> {'OK' if not_collapsed else 'COLLAPSED'}")
    print(f"  RESULT     : {'PASS' if result['passed'] else 'FAIL'}")
    return result


# --- hand-off artifact -------------------------------------------------------
def save_handoff(out_dir, model, scaler, reps, concepts, history, sanity, meta):
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. full model + scaler + the contract metadata the explainer needs.
    torch.save(
        {
            "state_dict": model.state_dict(),
            "scaler_mean": scaler.mean_.astype(np.float32).tolist(),
            "scaler_scale": scaler.scale_.astype(np.float32).tolist(),
            "raw_cols": INPUT_COLS,
            "d_in": int(model.d_in),
            "d_repr": int(model.d_repr),
            "arch": "input->256->128->32->1, ReLU",
        },
        out_dir / "stuff_mlp.pt",
    )

    # 2. the frozen readout head, loadable on its own: reps -> run value.
    torch.save(
        {
            "weight": model.head.weight.detach().cpu(),  # (1, d_repr)
            "bias": model.head.bias.detach().cpu(),       # (1,)
            "d_repr": int(model.d_repr),
        },
        out_dir / "readout_head.pt",
    )

    # 3. penultimate vectors, row-aligned with concepts.parquet.
    np.save(out_dir / "representations.npy", reps)

    # 4. concept columns, aligned row-for-row with the representations.
    concepts.to_parquet(out_dir / "concepts.parquet", index=False)

    # 5. a manifest documenting the contract (frames, shapes, defs).
    manifest = {
        "purpose": "Hand-off for the NLA explainer brief. The product is the "
                   "d_repr representation, aligned row-for-row with concepts.",
        "files": {
            "stuff_mlp.pt": "state_dict + scaler params + raw_cols + d_repr",
            "readout_head.pt": "standalone nn.Linear(d_repr,1): reps -> run value",
            "representations.npy": f"float32 {list(reps.shape)} penultimate vectors",
            "concepts.parquet": f"{concepts.shape[0]} rows x {concepts.shape[1]} cols, "
                                 "row-aligned with representations.npy",
        },
        "input_cols": INPUT_COLS,
        "concept_cols": CONCEPT_COLS,
        "frame": "All pitches mirrored into a unified RHP frame "
                 "(vx0, ax, release_pos_x, pfx_x negated for LHP; "
                 "spin_axis -> (360-spin_axis)%360).",
        "label": "run_value = -delta_run_exp (higher = better for pitcher)",
        "notes": {
            "axis_differential": "deg in [0,180] between spin-implied (spin_axis-90) "
                                 "and observed pfx movement direction; SSW proxy.",
            "arm_angle": "Statcast arm_angle where present, else geometric fallback.",
            "fb_is_fallback": "True where the pitcher-season had no FF/SI/FC and the "
                              "highest-usage pitch was used as the FB reference.",
        },
        "split": meta["split"],
        "range": {"start": meta["start"], "end": meta["end"]},
        "sanity": sanity,
        "final_val_huber": history["val"][-1] if history["val"] else None,
    }
    (out_dir / "handoff_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\nHand-off written to {out_dir}/")
    for f in ["stuff_mlp.pt", "readout_head.pt", "representations.npy",
              "concepts.parquet", "handoff_manifest.json"]:
        print(f"  - {f}")


# --- orchestration -----------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description="Train StuffMLP + write NLA hand-off.")
    p.add_argument("--full", action="store_true", help="2021-01-01 -> today (else 1-week smoke).")
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--split", choices=["random", "temporal"], default="random")
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--d-repr", type=int, default=32)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--huber-delta", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--max-rows", type=int, default=None,
                   help="Cap rows for the hand-off (samples reps+concepts together). "
                        "Useful for --full where all-row reps would be multi-GB.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    start, end = resolve_range(args.full, args.start, args.end)
    device = args.device
    print(f"Device: {device}  |  split: {args.split}  |  range: {start} -> {end}\n")

    # 1. data -> 2. features
    df = load_statcast(start, end, Path(args.cache_dir))
    X, y, concepts, ids = build_features(df)

    # standardize inputs (scaler fit on TRAIN ONLY - see below).
    Xv = X.to_numpy(dtype=np.float32)
    yv = y.to_numpy(dtype=np.float32)

    val_mask = make_split(ids, args.split, args.val_frac, args.seed)
    tr_mask = ~val_mask

    scaler = StandardScaler().fit(Xv[tr_mask])  # fit on train only -> no leakage
    Xtr = scaler.transform(Xv[tr_mask]).astype(np.float32)
    Xva = scaler.transform(Xv[val_mask]).astype(np.float32)
    ytr, yva = yv[tr_mask], yv[val_mask]
    print(f"  train {tr_mask.sum():,}  |  val {val_mask.sum():,}")

    # train
    print("\nTraining ...")
    model, history, best_val = train_model(
        Xtr, ytr, Xva, yva, d_repr=args.d_repr, epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay,
        huber_delta=args.huber_delta, patience=args.patience, device=device, seed=args.seed,
    )

    # representations for the FULL set, in input order (aligned with concepts/ids).
    X_all = scaler.transform(Xv).astype(np.float32)
    reps = extract_representations(model, X_all, device)
    concepts = concepts.copy()
    concepts.insert(0, "split", np.where(val_mask, "val", "train"))

    # optional row cap for the hand-off (sample reps+concepts identically).
    if args.max_rows is not None and len(reps) > args.max_rows:
        rng = np.random.default_rng(args.seed)
        keep = np.sort(rng.choice(len(reps), size=args.max_rows, replace=False))
        reps = reps[keep]
        concepts = concepts.iloc[keep].reset_index(drop=True)
        print(f"  capped hand-off to {args.max_rows:,} sampled rows")

    # sanity check + save. Representation-health is measured on the saved reps.
    sanity = sanity_check(model, Xva, yva, ytr, reps, args.huber_delta, device)
    save_handoff(
        Path(args.out_dir), model, scaler, reps, concepts, history, sanity,
        meta={"split": args.split, "start": start, "end": end},
    )

    print(f"\nDone. Exit criterion: {'PASS' if sanity['passed'] else 'FAIL'}.")


if __name__ == "__main__":
    main()
