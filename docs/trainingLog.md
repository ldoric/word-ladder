# Training Log

Everything we did for Word Ladder model training: data generation, RoBERTa fine-tuning, and results.

---

## Overview

**Goal:** Fine-tune RoBERTa to predict whether a **candidate** word is the correct next step on a shortest path from **current** to **target** (5-letter English Word Ladder).

**Task:** Binary classification  
- **Input:** `text_a` = "current [SEP] target", `text_b` = "candidate"  
- **Label:** 1 = correct next step, 0 = wrong  

**Pipeline:**  
1. Notebook 05: generate training data (CSVs, 50k cap)  
2. Notebook 06: fine-tune RoBERTa, evaluate, save model  

**Colab:** Run notebooks 05 and 06 on Google Colab (Runtime → GPU) for faster training. `models/`, `outputs/`, `data/training/*.csv` are gitignored — regenerate on Colab or upload.  

---

## 1. Data Generation (notebooks/05_english_5_letter_training.ipynb)

### Data sources
- **Playable words** (start/target): `data/islands/english_5_strict_largest_island.txt` (5330 words)
- **Full vocab** (allowed steps): `data/islands/english_5_largest_island.txt` (9902 words)

### Process
1. Build NetworkX graph on full vocab (one-letter edit distance)
2. Sample 18,000 random (start, target) pairs from playable
3. Find shortest paths; keep paths with **3–10 steps** (4–11 nodes)
4. For each path position: 1 positive (correct next step) + 3 negatives (wrong neighbors or random)
5. Deduplicate by (text_a, text_b, label)
6. Cap at 15,000 examples
7. Split by (start, target) to avoid leakage; stratify by path length (short/medium/long)
8. 90% train, 5% val, 5% test

### Output
- `data/training/wordladder_english5_train.csv` (~45k rows with 50k cap)
- `data/training/wordladder_english5_val.csv`
- `data/training/wordladder_english5_test.csv`

*Note: `models/`, `outputs/`, `data/training/*.csv` are in .gitignore — regenerate or upload for Colab.*

### Example row
```csv
text_a,text_b,label
sarin [SEP] sires,satin,0
saned [SEP] scrip,sayed,1
```

---

## 2. BERT Fine-Tuning (notebooks/06_bert_wordladder_finetune.ipynb)

### Model
- **Base:** `roberta-base` (upgraded from BERT)
- **Head:** `AutoModelForSequenceClassification` with 2 labels (binary)

### Hyperparameters
| Parameter | Value |
|-----------|-------|
| MAX_LENGTH | 64 tokens |
| BATCH_SIZE | 32 |
| EPOCHS | 3 |
| warmup_steps | 100 |
| weight_decay | 0.01 |
| SEED | 42 |

### Training setup
- HuggingFace `Trainer` with `TrainingArguments`
- AdamW optimizer
- Evaluation after each epoch
- Best model (by accuracy) loaded at end
- Saves to `outputs/bert_wordladder/` during training

### Output
- **Trained model:** `models/bert_wordladder_5letter/` (model + tokenizer)
- **Checkpoints:** `outputs/bert_wordladder/` (per-epoch checkpoints)

### Inference helper
`score_candidates(model, tokenizer, current, target, candidates)` – returns list of (candidate, score) sorted by P(label=1) descending.

---

## 3. Dependencies

```bash
pip install transformers torch pandas tqdm accelerate networkx
```

- **transformers** – BERT, tokenizer, Trainer
- **torch** – PyTorch
- **accelerate** – required by Trainer (device setup)
- **networkx** – used in notebook 05 for graph/path finding

---

## 4. How to Run

### Step 1: Generate data
1. Run `notebooks/04_english_datasets_testing.ipynb` to produce island files (if not present)
2. Run `notebooks/05_english_5_letter_training.ipynb` to generate CSVs

### Step 2: Fine-tune
1. Run `notebooks/06_bert_wordladder_finetune.ipynb`
2. Run the "Ensure accelerate" cell first if you get `ImportError` about accelerate
3. Let training complete (~3 hours on CPU, ~5–10 min on GPU)

### Step 3: Use the model
- Model saved to `models/bert_wordladder_5letter/`
- Load with `AutoModelForSequenceClassification.from_pretrained("models/bert_wordladder_5letter")`
- Use `score_candidates()` for inference

---

## 5. Hardware Notes

### CPU (laptop)
- **Time:** ~3 hours for 3 epochs (BERT, 13k examples, batch 32)
- **Note:** `pin_memory` warning is harmless on CPU
- **Device:** AMD Radeon laptops do not support PyTorch GPU (NVIDIA CUDA only)

### GPU (NVIDIA)
- **Time:** ~5–15 minutes for 3 epochs
- Install CUDA-enabled PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cu118`
- Trainer uses GPU automatically if available

### Google Colab
1. Upload/clone repo to Colab (or upload notebooks + data/islands/)
2. Runtime → Change runtime type → GPU (T4 free tier)
3. Run notebook 05 to generate CSVs (needs data/islands/)
4. Run notebook 06 to fine-tune (~10–15 min on GPU)
5. Download model from `models/` if needed

---

## 6. Troubleshooting

| Issue | Fix |
|-------|-----|
| `ImportError: accelerate>=1.1.0` | Run the "Ensure accelerate" cell in notebook 06; or `pip install accelerate` |
| `pin_memory` warning | Harmless on CPU; ignored |
| Training very slow | Use Colab GPU, or switch to `distilbert-base-uncased` |
| CUDA not available | PyTorch CPU-only; install CUDA build for NVIDIA GPU |

---

## 7. Results

### Run comparison

| Run | Model | Examples | Val Acc | Test Acc | Notes |
|-----|-------|----------|---------|----------|------|
| 1 (CPU) | bert-base-uncased | 15k | 77.45% | 79.51% | ~3h on CPU |
| 2 (Colab GPU) | roberta-base | 50k | 75.80% | 74.13% | same dataset |
| 3 (Colab GPU) | bert-base-uncased | 50k | **82.84%** | **81.75%** | same dataset as #2 |

### BERT Colab run (2025-03-20) — same dataset as RoBERTa
- **Training:** 3 epochs, batch 32, ~1480 s on Colab T4
- **Validation:** 82.84% | **Test:** 81.75%
- **Training loss:** 0.46 → 0.40 → 0.30 (still learning; val loss 0.45→0.44→0.46)
- **Inference demo:** lased→livid, correct `laved` ranked 6th (laser scored 0.89) — tricky case

### RoBERTa Colab run (2025-03-20)
- **Training:** 3 epochs, batch 32, ~1498 s (~25 min) on Colab T4
- **Validation:** 75.80%
- **Test:** 74.13%
- **Inference demo:** lased→livid, correct `laved` ranked 7th (tied at 0.216 with others)

RoBERTa performed **worse** than BERT on the **same** 50k dataset (75.8% vs 82.8% val). Likely reasons:
- BERT's native `[SEP]` matches our `current [SEP] target` format; RoBERTa uses `</s>` and may not align as well
- RoBERTa loss plateaued ~0.55; BERT kept improving (0.46→0.30 train loss)
- For this task and format, BERT fits better (Run 3 confirms)

### UNEXPECTED / MISSING keys when loading RoBERTa
Normal when loading `roberta-base` for sequence classification:
- **UNEXPECTED:** `lm_head.*`, `LayerNorm.beta/gamma` — RoBERTa uses different naming; can be ignored
- **MISSING:** `classifier.*` — new classification head; trained from scratch

### Path generation: model fails

When used **greedily** to generate full paths (start → target), the model **almost never succeeds** except for trivial 1-step cases (e.g. light→right). For any multi-step path it gets stuck; BFS fallback is required. **The current model is not useful for path generation.**

---

## 7b. Why the model fails at path generation (analysis)

### What we trained
- **Task:** Binary classification — given (current, target) + candidate, is the candidate the correct next step on a shortest path?
- **Result:** 82% test accuracy on held-out next-step classification.
- **Expected use:** Greedily pick best neighbor at each step to build a path.

### Why greedy path generation fails

1. **Compounding errors:** Even at 82% per-step accuracy, over 5 steps that’s ~0.82⁵ ≈ 35% chance of a correct full path. One wrong step can lead to a dead end and force a full restart.

2. **No graph awareness:** The model only sees text. It has no access to connectivity (which words can reach the target). It may rely on lexical similarity to the target (e.g. shared letters) instead of graph structure.

3. **Single positive per step:** We labeled only one “correct” next step per position. But many shortest paths exist; several neighbors can be valid. Training on a single positive may overfit to that choice and penalize others.

4. **Distribution shift:** Training used (current, target) pairs from true shortest paths. At inference, a single wrong step puts us in a state the model rarely saw: off-path. It may not generalize well from those states.

5. **Local vs global:** The model decides step-by-step without knowing the full path or remaining distance. It may prefer locally “good” moves that don’t lead to a valid path.

### Did we approach training wrong?

The setup (next-step binary classification) is reasonable, but it has limits:

- **Better targets:** Use all valid next steps on any shortest path as positives (multi-label), not just one.
- **Harder negatives:** Prefer wrong neighbors that look plausible (e.g. share letters with target) over random vocab words.
- **Alternative tasks:** Train a value function (estimated steps to target) and pick neighbors that minimize it; or use seq2seq to generate full paths; or use a graph-aware model (GNN) with connectivity.
- **Beam search:** Keep top-k candidates per step instead of pure greedy to reduce sensitivity to single mistakes.

### Bottom line

The model performs well on the classification task it was trained for, but that task does not transfer well to greedy path generation. The approach is sound in principle, but the current setup is not sufficient for reliable path finding.

---

## 8. Beam search (implemented)

Path generation now uses **beam search** (beam size 3 by default) instead of greedy. Try:

```bash
python scripts/play_wordladder.py saned scrip
python scripts/play_wordladder.py saned scrip --beam 5   # wider beam
python scripts/play_wordladder.py saned scrip --beam 1   # greedy
```

If beam search still fails often, use the multi-positive + harder negatives retrain below.

---

## 9. Multi-positive + harder negatives (for future retraining)

If beam search isn't enough, retrain with improved data. **Notebook 05** has flags (set both `True`):

- **MULTI_POSITIVE = True** — For each (current, target), label ALL neighbors that lie on any shortest path as positive (not just one). Lets the model accept multiple valid moves.
- **HARD_NEGATIVES = True** — Prefer wrong neighbors that share letters with the target (plausible but wrong) over random vocab words.

### What to run for training

1. **Notebook 05** — Set `MULTI_POSITIVE = True` and `HARD_NEGATIVES = True` at top of the "Create examples" cell. Run all cells. This generates new CSVs (overwrites `data/training/wordladder_english5_*.csv`).
2. **Notebook 06** — Run on Colab with GPU. Same as before; it reads the new CSVs.
3. Download the trained model and test with `play_wordladder.py`.

**Note:** With MULTI_POSITIVE, notebook 05 is slower (computes `shortest_path_length` per neighbor). Run on Colab if needed.

---

## 10. How to improve (training changes to try)

### Data (notebook 05)
| Change | Why |
|--------|-----|
| **More examples** | Raise cap from 15k to 30k–50k (you have ~400k before cap). More data → better generalization. |
| **More negatives per positive** | 4–5 instead of 3. Harder negatives improve discrimination. |
| **Hard negatives** | Prefer wrong neighbors that share letters with target (e.g. same position). Random-from-vocab is easier. |
| **Balance path lengths** | Oversample longer paths (7–10 steps) if underrepresented. |
| **Don’t cap** | Use all 400k examples if GPU allows. |

### Model & training (notebook 06)
| Change | Why |
|--------|-----|
| **More epochs** | Try 4–5. Loss may still be decreasing at epoch 3. |
| **Lower learning rate** | Default 5e-5 → try 2e-5 or 3e-5 for finer tuning. |
| **Larger batch size** | 64 if GPU memory allows. Can stabilize gradients. |
| **Different base model** | `roberta-base` or `deberta-v3-base` often outperform BERT on similar tasks. |
| **Learning rate schedule** | Try `cosine` instead of `linear` for warmup+decay. |
| **Label smoothing** | `label_smoothing_factor=0.1` can reduce overconfidence. |

### Input format
| Change | Why |
|--------|-----|
| **Add path position** | e.g. `current [SEP] target [SEP] step_2_of_6`. Gives context on remaining distance. |
| **Candidate position hint** | Encode which letter differs (e.g. “saned → sayed, position 2”). |
| **Multiple candidates per input** | Score 3–5 candidates in one forward pass (more efficient, possible comparison signal). |

### Evaluation
| Change | Why |
|--------|-----|
| **Top-k accuracy** | Report “correct in top-1, top-3, top-5” — more informative for ranking. |
| **Path-length breakdown** | Accuracy by 3–4 steps vs 7–10 steps. Longer paths may be harder. |
| **Hard subset** | Identify pairs where multiple valid next steps exist; track accuracy there. |

---

## 11. Changelog

- **2025-03-20:** Created training pipeline (05 data gen, 06 BERT finetune)
- **2025-03-20:** First CPU training run completed — val 77.45%, test 79.51%
- **2025-03-20:** Fixed evaluate cell (manual eval to avoid callback error)
- **2025-03-20:** Added improvement ideas (Section 8)
- **2025-03-20:** Raised cap to 50k, switched to roberta-base, added .gitignore for models/outputs/training CSVs
- **2025-03-20:** RoBERTa Colab run — val 75.80%, test 74.13% (worse than BERT baseline). Added colab-setup.md
- **2025-03-20:** BERT Colab run on same dataset — val 82.84%, test 81.75% (best so far)
- **2025-03-22:** Path generation testing: model fails for multi-step paths; only 1-step trivial cases work. BFS fallback added. Section 7b analysis added.
- **2025-03-22:** Beam search implemented (script + notebook). Multi-positive and harder negatives added to notebook 05 (flags off by default). Section 9 documents retraining steps.
