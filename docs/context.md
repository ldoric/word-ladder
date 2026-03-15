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

## Current counts (latest notebook run)
- english_5_strict.txt: 5811
- english_5.txt: 11155
- english_4_strict.txt: 3176
- english_4.txt: 5733
- croatian_*: run notebooks/02_croatian_wordlists.ipynb to generate

## Practical usage policy
- Start/target validation:
  - 5-letter tasks -> english_5_strict.txt (or croatian_5_strict.txt)
  - 4-letter tasks -> english_4_strict.txt (or croatian_4_strict.txt)
- Step expansion during solve:
  - 5-letter tasks -> english_5.txt (or croatian_5.txt)
  - 4-letter tasks -> english_4.txt (or croatian_4.txt)
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

## Caveats
- Counts can change if source files are updated upstream.
- If sources change, rerun generation and update this file plus docs/dataset-resources.md.
