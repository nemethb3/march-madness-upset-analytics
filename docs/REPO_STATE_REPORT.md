# REPO_STATE_REPORT

Generated: 2026-03-03

## Relevant Tree

### Top-level
- `.github/`
- `app/`
- `data/`
- `docs/`
- `notebooks/`
- `outputs/`
- `scripts/`
- `src/`
- `tools/`
- `README.md`
- `requirements.txt`
- `run_pipeline.py`
- `run_bracket_analysis.py`

### `app/`
- `app/streamlit_app.py`
- `app/components/bootstrap.py`
- `app/components/charts.py`
- `app/components/data_registry.py`
- `app/components/explanations.py`
- `app/components/io.py`
- `app/components/text.py`
- `app/pages/01_Upset_Alerts.py`
- `app/pages/02_Bracket_Builder.py`
- `app/pages/03_Simulations.py`

### `src/`
- `src/build_round1_from_slots.py`
- `src/inference_utils.py`
- `src/predict_matchups.py`
- `src/simulate_tournament.py`
- `src/upset_alerts.py`
- plus training/feature builders.

### `data/app/`
- `data/app/team_id_map.csv`
- `data/app/2025/seeds.csv`
- `data/app/2025/slots.csv`
- `data/app/2025/team_features.csv`
- `data/app/2026/seeds.csv`
- `data/app/2026/slots.csv`
- `data/app/2026/team_features.csv`
- `data/app/2026/README_demo.txt`

## Data Loading Functions
- `app/components/data_registry.py`
  - `list_available_seasons()`
  - `get_bundle_paths(season)`
  - `bundle_cache_token(season)`
  - `load_season_bundle(season)` (strict season-isolated load + schema validation)
- `app/components/io.py`
  - `cached_load_bundle(season, cache_token)` (`@st.cache_data`)
  - `render_sidebar()` (single source of truth season in `st.session_state["season"]`)
  - `build_round1_matchups_from_bracket(...)`
  - `score_matchups_df(...)`
  - `run_simulation_cached(...)` (`@st.cache_data`)

## Cache Usage
- `@st.cache_data`
  - `cached_load_bundle(season, cache_token)`
  - `run_simulation_cached(season, n_sims, randomness, model_hash, bundle_cache_token, ...)`
- `@st.cache_resource`
  - `load_cached_model(path_str)`

Cache keys now include season and bundle file token (`mtime_ns` + size), preventing cross-season stale reads.

## Module-Level DataFrames
- No global module-level dataframe constants found in `app/`.
- Dataframes are loaded in functions/pages from `ctx`.

## Session State Keys
- `season` (canonical selected season)
- `season_selector` (sidebar widget)
- `debug_mode`
- `alerts_selected_seed_pairs`
- `alerts_selected_levels`
- `alerts_show_all_games`
- `bracket_picks`
- `sim_adv_df`, `sim_matchup_df`, `sim_cache_season`

Season change callback clears dependent keys:
- alert widget selections
- bracket picks
- simulation cached outputs

## Hidden Fallback Logic
- No silent season fallback in sidebar loader.
- Missing/incomplete selected season bundle triggers explicit error + stop.
- Round 1 builder has explicit internal bracket fallback from seeds-only when slots cannot resolve full slate (intentional and deterministic).

