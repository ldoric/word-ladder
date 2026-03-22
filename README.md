# Word Ladder

A Python project for building and experimenting with word ladder data and logic, including BERT fine-tuning for next-step prediction.

## Current Status

- **Datasets:** English and Croatian 4/5-letter word lists (strict + non-strict, island-filtered)
- **Model:** RoBERTa fine-tuned on English 5-letter ladders (val ~77.5%, test ~79.5% baseline)
- **Model path:** `models/bert_wordladder_5letter/` (gitignored)

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `01_word_lists_combine.ipynb` | Build English word lists |
| `02_croatian_wordlists.ipynb` | Build Croatian word lists |
| `03_croatian_datasets_testing.ipynb` | Croatian connectivity + island extraction |
| `04_english_datasets_testing.ipynb` | English connectivity + island extraction |
| `05_english_5_letter_training.ipynb` | Generate BERT training data (CSVs) |
| `06_bert_wordladder_finetune.ipynb` | Fine-tune RoBERTa, evaluate, inference (run on Colab for GPU) |

## Docs

- **docs/context.md** — Agent context and dataset usage
- **docs/dataset-resources.md** — Source list and build formulas
- **docs/trainingLog.md** — Training pipeline, hyperparameters, results

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

```powershell
jupyter notebook notebooks/06_bert_wordladder_finetune.ipynb
```

Or run notebooks in order (01 → 04 → 05 → 06) for a full pipeline.
