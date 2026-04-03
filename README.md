# Word Ladder

A Python project for building and experimenting with word ladder data and logic, including BERT fine-tuned for **distance regression** (A\* heuristic on the word graph).

## Current Status

- **Datasets:** English and Croatian 4/5-letter word lists (strict + non-strict, island-filtered)
- **Model:** BERT distance regression (Run 7: ~600k examples, test MAE ~0.59, ~94% pure A\* on 200-pair eval — see `docs/trainingLog.md`)
- **Model path:** `models/bert_wordladder_5letter/` (gitignored)
- **Play:** `python scripts/play_wordladder.py START TARGET` — A\*-guided path between 5-letter words

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `01_word_lists_combine.ipynb` | Build English word lists |
| `02_croatian_wordlists.ipynb` | Build Croatian word lists |
| `03_croatian_datasets_testing.ipynb` | Croatian connectivity + island extraction |
| `04_english_datasets_testing.ipynb` | English connectivity + island extraction |
| `05_english_5_letter_training.ipynb` | Generate BERT training data (CSVs) |
| `06_bert_wordladder_finetune.ipynb` | Fine-tune BERT, evaluate, inference, path generation (run on Colab for GPU) |

## Docs

- **docs/context.md** — Agent context and dataset usage
- **docs/dataset-resources.md** — Source list and build formulas
- **docs/trainingLog.md** — Training pipeline, hyperparameters, results
- **docs/colab-setup.md** — Google Colab setup and model download

## Project Setup

1. Create and activate a virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
2. Install dependencies:
   ```powershell
   pip install jupyter transformers torch pandas tqdm accelerate networkx
   ```

## Run

**Train (on Colab):** See `docs/colab-setup.md`.

**Play with a downloaded model:**
```powershell
python scripts/play_wordladder.py crane flame
python scripts/play_wordladder.py   # interactive mode
```

**Notebooks:** Run in order 01 → 04 → 05 → 06. Or open `06_bert_wordladder_finetune.ipynb` and use the "Generate path" section.
