# APP STATE + FIX Audit Report

Generated on: 2026-03-03 (America/New_York)

## 1) Current Project Structure

### Top-level tree
```text
march-madness-upset-analytics/
├─ app/
├─ data/
├─ notebooks/
├─ outputs/
├─ src/
├─ tools/
├─ .gitignore
├─ analysis.ipynb
├─ README.md
├─ requirements.txt
├─ run_bracket_analysis.py
└─ run_pipeline.py
```

### `app/` tree
```text
app/
├─ streamlit_app.py
├─ __init__.py
├─ components/
│  ├─ bootstrap.py
│  ├─ charts.py
│  ├─ data_registry.py
│  ├─ explanations.py
│  ├─ io.py
│  ├─ text.py
│  └─ __init__.py
├─ demo_data/
│  ├─ demo_seeds.csv
│  ├─ demo_slots.csv
│  └─ demo_team_features.csv
└─ pages/
   ├─ 01_Upset_Alerts.py
   ├─ 02_Bracket_Builder.py
   └─ 03_Simulations.py
```

### `src/` tree
```text
src/
├─ build_advanced_team_season_features.py
├─ build_conference_features.py
├─ build_massey_features.py
├─ build_round1_from_slots.py
├─ build_team_season_features.py
├─ build_tourney_matchups.py
├─ config.py
├─ evaluate.py
├─ historical_upset_rates.py
├─ inference_utils.py
├─ io_utils.py
├─ predict_matchups.py
├─ simulate_tournament.py
├─ train_models.py
├─ upset_alerts.py
└─ __init__.py
```

### `data/app/` tree
```text
data/app/
├─ team_id_map.csv
├─ 2025/
│  ├─ seeds.csv
│  ├─ slots.csv
│  └─ team_features.csv
└─ 2026/
   ├─ README_demo.txt
   ├─ seeds.csv
   ├─ slots.csv
   └─ team_features.csv
```

## 2) Bundle Inventory and Required File Check

Required files per season bundle:
- `seeds.csv`
- `slots.csv`
- `team_features.csv`
- global: `data/app/team_id_map.csv`

Detected seasons under `data/app/`:
- `2025`
- `2026`

### Season 2025
- Present files: `seeds.csv`, `slots.csv`, `team_features.csv`
- Required file presence: PASS
- Bracket readiness: FAIL (only partial demo; does not support full Round 1 slate)

### Season 2026
- Present files: `seeds.csv`, `slots.csv`, `team_features.csv`, `README_demo.txt`
- Required file presence: PASS
- Bracket readiness: PASS

## 3) 2026 Bundle Validation Details

- `seeds.csv` rows: **64**
- unique `TeamID` in `seeds.csv`: **64**
- `team_features.csv` rows: **64**
- unique `TeamID` in `team_features.csv`: **64**
- `slots.csv` rows: **63**
- Round 1 resolvable games from slots+seeds: **32**
- First Four support: **No**

### First Four banner
The 2026 demo bundle is currently a **64-team synthetic bracket** (no First Four).  
Reason: local source slot structure available for generation was incomplete (`data/app/2025/slots.csv` had only 7 rows), and `data/raw/MNCAATourneySlots.csv` was not present locally. Generator now falls back to a complete 63-slot standard bracket and documents this in `data/app/2026/README_demo.txt`.

## 4) Exact Validation Command Output (2026)

Command:
```bash
python tools/validate_bundle.py --season 2026
```

Output:
```text
VALIDATE BUNDLE season=2026
app_dir=data\app

[PASS] file:seeds path=data\app\2026\seeds.csv
[PASS] file:slots path=data\app\2026\slots.csv
[PASS] file:team_features path=data\app\2026\team_features.csv
[PASS] file:team_id_map path=data\app\team_id_map.csv
[PASS] schema:seeds.csv missing=[]
[PASS] schema:slots.csv missing=[]
[PASS] schema:team_features.csv missing=[]
[PASS] schema:team_id_map.csv missing=[]

counts.seeds_rows=64
counts.seeds_unique_teams=64
counts.team_features_rows=64
counts.team_features_unique_teams=64
counts.slots_rows=63
counts.team_id_map_rows=381
round1.total_seed_seed_slots=32
round1.resolved_games=32
round1.seed_pair_types=['1 vs 16', '2 vs 15', '3 vs 14', '4 vs 13', '5 vs 12', '6 vs 11', '7 vs 10', '8 vs 9']
first_four_supported=False
[PASS] coverage.seeded_teams_in_features missing_count=0
coverage.extra_feature_team_ids_count=0
[PASS] round1 resolved games >= 32

RESULT: PASS
```

## 5) What Was Wrong, What Changed, What Is Now True

### What was wrong
- Demo source bundle for 2025 was incomplete (8 seeds, 7 slots), so deriving 2026 from it could be misleading.
- There was no dedicated bundle validator script for quick PASS/FAIL readiness checks.
- App sidebar logic could silently fall back to another season on bundle errors, which hides missing-file issues for selected season.

### What I changed
- Added `tools/validate_bundle.py` to validate required files, schema, row counts, team-feature coverage, Round 1 resolvability, and First Four support.
- Reworked `tools/generate_demo_season.py`:
  - attempts to copy full slot structure from raw first, then bundle source.
  - if source slots are incomplete, falls back to a complete 63-slot 64-team bracket.
  - generates deterministic synthetic seeds/features with `--seed`.
  - enforces 1:1 feature coverage for seeded teams.
  - writes `data/app/2026/README_demo.txt` with generation mode details.
- Updated `app/components/io.py` sidebar loading behavior:
  - selected season bundle errors now show a friendly error and `st.stop()` (no silent fallback).

### What is now true
- `data/app/2026` is complete and bracket-ready for app use.
- 2026 validates with PASS and supports a full 32-game Round 1 slate.
- Bundle health can now be checked reproducibly via:
  - `python tools/validate_bundle.py --season 2026`
