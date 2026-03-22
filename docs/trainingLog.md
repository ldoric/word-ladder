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

## 7. Results (baseline setup)

| Split | Accuracy |
|-------|----------|
| Validation | 77.45% |
| Test | 79.51% |

**Baseline (BERT):** bert-base-uncased, 3 epochs, batch 32, 15k examples, ~3h on CPU.

**Current setup:** roberta-base, 50k examples, run on Colab GPU.

**Inference:** Correct next step often in top-2 or top-3 (e.g. saned→scrip: sayed ranked 2nd behind sanes). Model is useful even when not #1.

---

## 8. How to improve (training changes to try)

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

## 9. Changelog

- **2025-03-20:** Created training pipeline (05 data gen, 06 BERT finetune)
- **2025-03-20:** First CPU training run completed — val 77.45%, test 79.51%
- **2025-03-20:** Fixed evaluate cell (manual eval to avoid callback error)
- **2025-03-20:** Added improvement ideas (Section 8)
- **2025-03-20:** Raised cap to 50k, switched to roberta-base, added .gitignore for models/outputs/training CSVs
