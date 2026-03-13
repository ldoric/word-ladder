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

## Current counts (latest notebook run)
- english_5_strict.txt: 5811
- english_5.txt: 11155
- english_4_strict.txt: 3176
- english_4.txt: 5733

## Practical usage policy
- Start/target validation:
  - 5-letter tasks -> english_5_strict.txt
  - 4-letter tasks -> english_4_strict.txt
- Step expansion during solve:
  - 5-letter tasks -> english_5.txt
  - 4-letter tasks -> english_4.txt
- 4-letter construction detail:
  - strict: words appearing in at least 2 curated 4-letter sources
  - non-strict: union of curated 4-letter sources
- Final path check:
  - Every token in the produced path must exist in the non-strict set used for expansion.
  - Optionally report whether each token is also in the strict set (for analysis).

## Why this split helps
- Strict sets reduce noisy edge cases and improve benchmark fairness.
- Non-strict sets increase graph connectivity, improving success rate on hard ladders.
- This aligns with thesis goal: compare rule adherence and reliability across methods.

## Reproducibility note
- Main generation logic currently lives in Notebook 1, Cell 5:
  - notebooks/01_word_lists_combine.ipynb
- Re-run that cell to regenerate all dataset files in data/.

## Caveats
- Counts can change if source files are updated upstream.
- If sources change, rerun generation and update this file plus docs/dataset-resources.md.
