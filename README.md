# Why+ — the Stuff+ model

A **from-scratch Stuff+**: an MLP that predicts a pitch's **run value** from raw
pitch **physics**. Predicting run value from physical characteristics *is* the
Stuff+ method — this is our own, not a wrapper around anyone else's.

The model is a *means*, not the end. Its job is to carry a 32-d penultimate
**representation worth explaining later**. **Do not optimize for predictive
accuracy** — low per-pitch val R² (~0.05–0.15) is the noise floor, not failure.

This repo is the `model/` brief only. The explainer (verbalizer + reconstruction
loop) and the concept-probe/gate are a **separate NLA brief** that consumes the
hand-off artifact produced here.

## Layout

```
whyplus/model/
  data.py      # pull + cache Statcast (per-season parquet)
  features.py  # LHP mirror, fastball context, concept columns
  net.py       # StuffMLP -> (pred, penultimate)
  train.py     # train; save hand-off artifact; non-degeneracy sanity check
```

## Setup

Uses [`uv`](https://docs.astral.sh/uv/). torch is pinned to the **cu128** wheels
(works on this box's CUDA-13 driver / Ampere 3090s; the net is tiny so CPU also
finishes, but the explainer phase needs the GPU).

```bash
uv sync
```

## Run

```bash
# 1-week smoke window (~50k pitches) — end-to-end cheap, also an egress probe.
uv run python -m whyplus.model.train

# full pull (2021-01-01 -> today) then train. Pulls are slow & cached per season.
uv run python -m whyplus.model.train --full

# just do the (slow) pull+cache step:
uv run python -m whyplus.model.data --full
```

Useful flags: `--split {random,temporal}` (default `random`; temporal is
train ≤2024 / val 2025), `--max-rows N` (cap the hand-off for `--full` so
`representations.npy` stays small), `--start/--end YYYY-MM-DD`.

## Hand-off artifact (the contract the NLA brief consumes)

Written to `artifacts/`:

| file | what |
|------|------|
| `stuff_mlp.pt` | `state_dict` + scaler params + `raw_cols` + `d_repr` |
| `readout_head.pt` | frozen `nn.Linear(d_repr, 1)`, loadable on its own |
| `representations.npy` | `(N, 32)` penultimate vectors |
| `concepts.parquet` | concept columns, **row-aligned** with the representations |
| `handoff_manifest.json` | frames, shapes, concept definitions |

### Conventions

- **Frame:** all pitches mirrored into a unified RHP frame (`vx0, ax,
  release_pos_x, pfx_x` negated for LHP; `spin_axis → (360-spin_axis)%360`).
- **Inputs (~24 dims):** the 12 raw constants + their `fb_*` fastball-context
  means. **No `pfx_*`, no location, no count, no batter handedness** — *stuff* only.
- **Label:** `run_value = -delta_run_exp`.
- **Concept columns** (computed here, *probed later*): `pfx_x, pfx_z, vaa,
  release_extension, arm_angle, axis_differential`.

## Decisions (v0)

- Random 85/15 split. (Temporal available via `--split temporal`.)
- Fastball-less pitcher-seasons use the highest-usage pitch as the reference,
  flagged in `concepts.fb_is_fallback`.
- `axis_differential` is computed now (spin-implied vs observed movement angle).

**Commit code, not data.** `data/` and `artifacts/` are gitignored.
