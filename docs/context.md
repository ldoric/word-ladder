# Context for AI Agents

This file is a working memory reference for agents operating on this repo.

## Dataset roles
- Use strict datasets for evaluation endpoints (start/target words).
- Use non-strict datasets for search expansion (intermediate steps) to improve path finding coverage.

## Current dataset files
- data/english_5_strict.txt
- data/english_5.txt
- data/english_4_strict.txt
- data/english_4.txt
- data/croatian_5_strict.txt
- data/croatian_5.txt
- data/croatian_4_strict.txt
- data/croatian_4.txt
- data/islands/english_*_largest_island.txt, croatian_*_largest_island.txt (non-strict, largest component only)
- data/islands/english_*_strict_largest_island.txt, croatian_*_strict_largest_island.txt (strict endpoints, non-strict steps)
- data/islands/english_*_strict_only_island.txt, croatian_*_strict_only_island.txt (strict steps and endpoints)

## Current counts (latest notebook run)
- english_5_strict.txt: 5811
- english_5.txt: 11155
- english_4_strict.txt: 3176
- english_4.txt: 5733
- croatian_*: run notebooks/02_croatian_wordlists.ipynb to generate
- croatian_4_strict: manually curated (complete)
- croatian_5_strict: manually curated via helper cells (e.g. h-ending glagol forms)
- english island files: 4-letter largest 5643, strict_largest 3155, strict_only 3050; 5-letter largest 9902, strict_largest 5330, strict_only 4573
- croatian island files: 4-letter largest 3863, strict_largest 941, strict_only 767; 5-letter largest 11052, strict_largest 1448, strict_only 57

## Practical usage policy
- Start/target validation:
  - 5-letter tasks -> english_5_strict.txt (or croatian_5_strict.txt)
  - 4-letter tasks -> english_4_strict.txt (or croatian_4_strict.txt)
- Step expansion during solve:
  - 5-letter tasks -> english_5.txt (or croatian_5.txt)
  - 4-letter tasks -> english_4.txt (or croatian_4.txt)
- Gameplay (guaranteed solvable pairs): use data/islands/ files. largest_island = non-strict; strict_largest_island = strict endpoints + non-strict steps; strict_only_island = strict only. English 5-letter strict_only 4573; Croatian 5-letter strict_only 57.
- English 4-letter: strict = ≥2 curated sources, non-strict = union
- Croatian: strict = all 3 sources (Rijecalica ∩ kkrypt0nn ∩ HR_Txt), non-strict = ≥2 sources
- Final path check:
  - Every token in the produced path must exist in the non-strict set used for expansion.
  - Optionally report whether each token is also in the strict set (for analysis).

## Why this split helps
- Strict sets reduce noisy edge cases and improve benchmark fairness.
- Non-strict sets increase graph connectivity, improving success rate on hard ladders.
- This aligns with thesis goal: compare rule adherence and reliability across methods.

## Reproducibility note
- English: notebooks/01_word_lists_combine.ipynb (Cell 4)
- Croatian: notebooks/02_croatian_wordlists.ipynb (Cell 5 build, Cell 7 review gen, Cell 9 apply removals)
- English islands: notebooks/04_english_datasets_testing.ipynb (filter to largest component)
- Croatian islands: notebooks/03_croatian_datasets_testing.ipynb (filter to largest component)
- BERT training data (English 5-letter): notebooks/05_english_5_letter_training.ipynb (distance regression CSVs: wordladder_english5_*.csv)
- BERT training data (English 4-letter): notebooks/07_english_4_letter_training.ipynb (distance regression CSVs: wordladder_english4_*.csv)
- BERT training data (Croatian 5-letter): notebooks/09_croatian_5_letter_training.ipynb (distance regression CSVs: wordladder_croatian5_*.csv)
- BERT training data (Croatian 4-letter): notebooks/11_croatian_4_letter_training.ipynb (distance regression CSVs: wordladder_croatian4_*.csv)
- LLM vs BERT benchmark: notebooks/13_llm_vs_bert_wordladder.ipynb — set `GRAPH_PRESET`, `GPT_MODEL`, optional Gemini; `EVAL_SET_NAME=None` auto-picks `data/eval_sets/<name>.csv` per preset (`english_4` → `test4english`, etc.) and creates it if missing; API keys in `.env.local` — see docs/colab-setup.md
- Deployed hint API smoke test (four `POST /predict`): notebooks/15_model_api_test.ipynb (`WORD_LADDER_API_BASE` optional; default `https://ldoric-word-ladder-api.hf.space`)
- LLM comparison log (benchmark results): docs/comparing-to-llms.md

## BERT hint API (Hugging Face Space)

Production **FastAPI** service for “next step” hints: scores one-letter neighbors with the same distance-regression BERT as `scripts/play_wordladder.py` (`score_candidates`). Code and Docker layout: **`word-ladder-api/`** (see `word-ladder-api/README.md` for deploy, env vars, GitHub vs Hub).

### URLs (important)

- **Call the API on the direct host:** `https://<username>-<spacename>.hf.space`  
  Example (current): **`https://ldoric-word-ladder-api.hf.space`** — endpoints: `GET /`, `GET /health`, `GET /docs`, `POST /predict`.
- **Do not** use `https://huggingface.co/spaces/<user>/<spacename>/...` as the HTTP base for `/health` or `/predict` — that site is the Space *page*; path routing often returns **404** for API routes. The running app is on **`.hf.space`**.

**Space (git) repo:** `https://huggingface.co/spaces/ldoric/word-ladder-api` (push with HF token, not account password).

### Runtime configuration (on the Space)

- **Model repo ids (Variables):** `HF_HUB_EN_4`, `HF_HUB_EN_5`, `HF_HUB_HR_4`, `HF_HUB_HR_5` = Hub model id per mode (e.g. `ldoric/bert-wl-en5` for `en_5`).
- **Private models (Secrets):** `HUGGING_FACE_HUB_TOKEN` or `HF_TOKEN` (read) so `download_models_from_hub.py` can pull weights at container start.
- **Optional:** `PRELOAD_ALL_MODELS=1` to load all four BERTs at startup (needs RAM).
- **All four modes** (`en_4`, `en_5`, `hr_4`, `hr_5`) are live once each `HF_HUB_*` points at a Hub model with weights (private repos need the token secret). Smoke test: **`notebooks/15_model_api_test.ipynb`** (one `POST /predict` per mode).

### `POST /predict` contract

- **Body (JSON):** `current`, `target`, `mode` ∈ `en_4` | `en_5` | `hr_4` | `hr_5`, optional `full_ranking`.
- **Response:** `best_neighbor`, `predicted_distance` (lower = better distance estimate to `target`); 400 if words missing from the API’s word list.
- **Word lists** inside the service mirror **`data/islands/*_largest_island.txt`** (copies under `word-ladder-api/data/words_*.txt`).

## Caveats
- Counts can change if source files are updated upstream.
- If sources change, rerun generation and update this file plus docs/dataset-resources.md.
