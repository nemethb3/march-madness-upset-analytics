# APP_AUDIT_REPORT

Generated: 2026-03-03

## Scope
Audit focused on season switching, bundle loading, Upset Alerts data flow, Bracket Builder state, and simulation state isolation.

## Findings

### 1) Season switch state contamination existed
- `Season` selector was not managed as a single canonical `st.session_state["season"]`.
- Dependent page states (`bracket_picks`, `sim_adv_df`, `sim_matchup_df`, alert filter widget state) were not reset on season change.
- Result: stale outputs could persist across season changes.

### 2) Cache keys were incomplete for data freshness
- Bundle cache previously keyed only by `season`.
- If season files changed in-place, cache could remain stale.

### 3) Simulation page stale display risk
- Simulation results were stored in session state without season tagging.
- Switching season could show previous season simulation outputs until rerun.

### 4) Upset Alerts normalization/filtering needed hardening
- NaN-safe normalization was added earlier, but this audit confirmed filter/state behavior needed tighter season isolation.
- Filter options are now derived from full season-scoped dataset and no longer from filtered views.

## Changes Applied

### Centralized, season-isolated bundle loading
- Updated [`app/components/data_registry.py`](/c:/Users/brend/Downloads/March_ML_Mania_2026/march-madness-upset-analytics/app/components/data_registry.py):
  - Added `get_bundle_paths(season)`.
  - Added `bundle_cache_token(season)` based on file mtimes/sizes.
  - Hardened `load_season_bundle(season)`:
    - strict required file checks
    - schema validation
    - season column enforcement/filtering to selected season only
    - type coercion for TeamID fields
    - deterministic deduping

### Cache key fixes
- Updated [`app/components/io.py`](/c:/Users/brend/Downloads/March_ML_Mania_2026/march-madness-upset-analytics/app/components/io.py):
  - `cached_load_bundle(season, cache_token)` now takes season + bundle token.
  - `run_simulation_cached(...)` now includes `bundle_cache_token` in signature.

### Single source of truth for season + reset on season change
- Updated [`app/components/io.py`](/c:/Users/brend/Downloads/March_ML_Mania_2026/march-madness-upset-analytics/app/components/io.py):
  - season selector uses `st.session_state["season_selector"]`.
  - canonical season set to `st.session_state["season"]`.
  - on-change callback clears dependent state:
    - `alerts_selected_seed_pairs`
    - `alerts_selected_levels`
    - `alerts_show_all_games`
    - `bracket_picks`
    - `sim_adv_df`, `sim_matchup_df`, `sim_cache_season`

### Debug mode + deterministic diagnostics
- Added sidebar `Debug mode` checkbox in `render_sidebar`.
- Debug expanders on Upset Alerts, Bracket Builder, Simulations are shown only in debug mode.

### Upset Alerts flow hardened
- Updated [`app/pages/01_Upset_Alerts.py`](/c:/Users/brend/Downloads/March_ML_Mania_2026/march-madness-upset-analytics/app/pages/01_Upset_Alerts.py):
  - canonical full dataframe (`full_df`) for filter options/charts.
  - filtered dataframe (`view_df`) for display only.
  - seed options always from `full_df["SeedPair"]`.
  - robust empty-state explanation with counts at each filter stage (debug mode).
  - NaN-safe seed normalization retained.

### Bracket/Sim pages season consistency
- Updated [`app/pages/02_Bracket_Builder.py`](/c:/Users/brend/Downloads/March_ML_Mania_2026/march-madness-upset-analytics/app/pages/02_Bracket_Builder.py): debug expander for season-specific shapes.
- Updated [`app/pages/03_Simulations.py`](/c:/Users/brend/Downloads/March_ML_Mania_2026/march-madness-upset-analytics/app/pages/03_Simulations.py):
  - pass `bundle_cache_token` to cached simulation function.
  - clear simulation outputs when `sim_cache_season != current season`.

### Reason quality
- Updated [`app/components/explanations.py`](/c:/Users/brend/Downloads/March_ML_Mania_2026/march-madness-upset-analytics/app/components/explanations.py):
  - reasons now include numeric delta signal in text, improving per-matchup variation.

## Validation Evidence

Command:
```bash
python scripts/validate_app_state.py --season 2026 --check_secondary
```

Output summary:
- 2026:
  - seeds unique teams: 64 (PASS)
  - round1 resolved games: 32 (PASS)
  - alerts required columns: PASS
  - seed pair options count: 8 (PASS)
  - reason diversity: 32 unique sets (PASS)
- 2025:
  - partial bundle warnings (8-team demo), no crashes.

Additional direct check:
- 2026 full Round 1 scored rows: 32
- seed pairs present: all 8 (`1 vs 16` ... `8 vs 9`)
- threshold filtering changes `view_df` size while options remain complete.

## Residual Notes
- `data/app/2025` remains intentionally partial demo data; season switching is stable, but 2025 has fewer matchups by design.
- 2026 demo bundle is full bracket-ready and validated.
