# FIX_REPORT

Generated: 2026-03-03

## Problem Reproduction (Before Fix)

Symptoms reported:
- Switching `Season` (2025 ↔ 2026) did not reliably refresh page state.
- Upset Alerts and filters appeared stale.
- Simulations and bracket state could persist across season changes.

Root causes identified in code:
1. No canonical `st.session_state["season"]` source of truth.
2. No season-change callback to clear dependent session state.
3. Cached bundle loading keyed only by `season`, not by underlying file change token.
4. Simulation outputs persisted in session state without season tagging.
5. Alerts filter widgets could persist stale selections unless reset.

## What Was Changed

### Data loading and cache hardening
- Added season-path and cache token support in `app/components/data_registry.py`:
  - `get_bundle_paths(season)`
  - `bundle_cache_token(season)`
  - stricter `load_season_bundle(season)` season isolation and schema/type validation.
- Updated cached loader in `app/components/io.py`:
  - `cached_load_bundle(season, cache_token)`.

### Season state unification and reset
- Season selector now writes to `st.session_state["season"]`.
- Added season change callback in `app/components/io.py` that clears dependent keys:
  - alerts filters
  - bracket picks
  - simulation outputs.

### Upset Alerts pipeline stability
- Kept deterministic full dataset (`full_df`) separate from filtered display (`view_df`).
- Seed pair options always derived from `full_df`.
- Added debug counts for each filter stage (in debug mode).
- NaN-safe normalization retained.

### Bracket + Sim pages season integrity
- Sim page now invalidates stale cached outputs when season changes.
- Sim cache signature includes `bundle_cache_token`.
- Added debug expanders for data path/shape validation (debug mode only).

### Explanations
- Reason strings now include numeric delta signal (more matchup-specific, less boilerplate).

## How to Validate (After Fix)

1. Run static validation:
```bash
python scripts/validate_app_state.py --season 2026 --check_secondary
```

2. Run app:
```bash
streamlit run app/streamlit_app.py
```

3. In sidebar:
- Toggle between `2025` and `2026`.
- Verify `Backend Status` season/shape changes.
- Enable `Debug mode` and verify:
  - Upset Alerts `full_df_shape` tracks season.
  - `seed_pair_options` remain season-accurate.

4. Upset Alerts checks:
- 2026 should show full Round 1 slate in source data (`full_df` ~32 rows).
- Seed pair filter options should include all 8 pairs.
- Changing threshold/levels changes `view_df` only, not options.

## Validation Evidence Captured

From `python scripts/validate_app_state.py --season 2026 --check_secondary`:
- 2026:
  - Round 1 resolved games: 32
  - Seed pair options: 8
  - Reason set diversity: 32 unique sets
- 2025:
  - partial bundle warnings only; pipeline remains robust and non-crashing.
