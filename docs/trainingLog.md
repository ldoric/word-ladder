# Training Log

Everything we did for Word Ladder model training: data generation, BERT fine-tuning, and results.

---

## Overview

**Goal:** Fine-tune BERT to predict the **BFS shortest-path distance** between two words in the word-ladder graph (5-letter English). Use this as an A\* heuristic for path generation.

**Task:** Regression (MSE loss)  
- **Input:** `text_a` = word\_a, `text_b` = word\_b (BERT sentence pair)  
- **Label:** shortest-path distance (integer, e.g. 3)  

**Pipeline:**  
1. Notebook 05: generate distance-regression training data (CSVs) — 5-letter English  
2. Notebook 06: fine-tune BERT, evaluate, save model — 5-letter English  
3. Notebook 07: generate distance-regression training data (CSVs) — 4-letter English  
4. Notebook 08: fine-tune BERT, evaluate, save model — 4-letter English  
5. Notebook 09: generate distance-regression training data (CSVs) — 5-letter Croatian  
6. Notebook 10: fine-tune BERT, evaluate, save model — 5-letter Croatian  
7. Notebook 11: generate distance-regression training data (CSVs) — 4-letter Croatian  
8. Notebook 12: fine-tune BERT, evaluate, save model — 4-letter Croatian  

**Colab:** Run notebooks 05 and 06 on Google Colab (Runtime → GPU) for faster training. `models/`, `outputs/`, `data/training/*.csv` are gitignored — regenerate on Colab or upload.  

**Run 7+ notebook defaults (scale-up):**

| Setting | Notebook 05 | Notebook 06 |
|--------|-------------|-------------|
| Data size | **600k** examples | — |
| BFS sources | **4000** | — |
| Sampling | **Weighted** (higher weight for graph distance ≤3, then ≤6, then ≤9) | — |
| Training | — | **6 epochs**, **cosine** LR schedule, **warmup_ratio=0.06** |
| Logging | — | `logging_steps=100` |

**Run 7 trained model:** Use CSVs from notebook 05 (600k + weighted sampling) + checkpoint from notebook 06 as above. To **reproduce** Run 7, run **05 → 06** again; older **250k** checkpoints are superseded.

---

## Phase 1: Binary Classification (Runs 1–4, archived)

The original approach trained a binary classifier: given `(current, target)` + `candidate`, predict whether the candidate is the correct next step on a shortest path. This reached 82% accuracy (Run 3) but **failed at path generation** — the model could not reliably build multi-step paths.

### Why it failed

1. **Compounding errors:** 82% per-step accuracy → ~35% over 5 steps.
2. **No graph awareness:** The model saw text, not graph connectivity. It learned lexical similarity, not distance structure.
3. **Poor ranking calibration:** Binary classification probabilities (P(correct)) are not calibrated for ranking neighbors. A 0.73 vs 0.71 difference is noise, not signal.
4. **Distribution shift:** One wrong step puts the model in a state it never trained on (off-path).
5. **Positive/negative imbalance:** 25% positive / 75% negative — model could get ~75% by always predicting "no."

### Phase 1 run comparison

| Run | Model | Examples | Val Acc | Test Acc | Notes |
|-----|-------|----------|---------|----------|------|
| 1 (CPU) | bert-base-uncased | 15k | 77.45% | 79.51% | ~3h on CPU |
| 2 (Colab GPU) | roberta-base | 50k | 75.80% | 74.13% | same dataset |
| 3 (Colab GPU) | bert-base-uncased | 50k | **82.84%** | **81.75%** | single-positive (best classification) |
| 4 (Colab GPU) | bert-base-uncased | 50k | 73.07% | 73.40% | multi-pos + hard neg — worse |

### Phase 1 conclusions

- BERT outperformed RoBERTa on this task (82.8% vs 75.8%)
- Multi-positive + hard negatives made things worse (73.4% vs 81.8%)
- Path generation failed for all variants: beam search, greedy, multi-pos — always fell back to BFS
- **The classification task does not transfer to path generation.** The approach is fundamentally misaligned with the use case.

---

## Phase 2: Distance Regression (current approach)

### Why we switched

Binary classification answers "is this one step correct?" — but path generation needs "which neighbor is closest to the target?" Distance regression directly predicts what A\* search needs: the shortest-path distance between any two words. At inference we score all neighbors and pick the one with the **lowest predicted distance**.

Key advantages:
- **Aligned objective:** The model learns exactly what the search algorithm uses — distance estimation.
- **More training data:** Every reachable (word\_a, word\_b) pair has a distance label. No need for positive/negative construction.
- **No class imbalance:** Regression on continuous values, not 25/75 binary split.
- **Better ranking:** Distance predictions are naturally ordinal — a prediction of 3.1 vs 5.2 is a confident ranking. Classification probabilities like 0.73 vs 0.71 are not.
- **Graph structure:** The model must learn global topology (dense vs sparse regions) to predict distances accurately.

### Data generation (notebook 05)

1. Build NetworkX graph on full vocab (9,902 words, 34k edges)
2. Run single-source BFS from **N** random words → collect pairwise distances (Run 7+: **N = 4000**)
3. Deduplicate symmetric pairs (dist(a,b) = dist(b,a))
4. Subsample to **TARGET_EXAMPLES** with **weighted** distance coverage (Run 7+: **600k** rows; higher weight for short graph distances d≤3, ≤6, ≤9 — better for A\* neighbor ranking)
5. Randomly swap word order (model learns symmetry)
6. Split by pair; stratify by distance bin (short 1–3, medium 4–6, long 7+)
7. 90% train, 5% val, 5% test

### Example row
```csv
text_a,text_b,label
crane,flame,3
stone,heart,7
```

### Model

- **Base:** `bert-base-uncased`
- **Head:** `AutoModelForSequenceClassification` with `num_labels=1, problem_type="regression"`
- **Loss:** MSE (handled automatically by HuggingFace Trainer)

### Hyperparameters

| Parameter | Value (Run 7+) |
|-----------|----------------|
| MAX_LENGTH | 32 tokens |
| BATCH_SIZE | 32 (use 16 if Colab OOM) |
| EPOCHS | **6** |
| learning_rate | 2e-5 |
| LR schedule | **cosine** |
| warmup | **warmup_ratio=0.06** (replaces fixed warmup_steps) |
| weight_decay | 0.01 |
| SEED | 42 |

### Metrics

- **MAE** (Mean Absolute Error): average prediction error in steps
- **RMSE** (Root Mean Squared Error): penalizes large errors more
- **Within ±1 step**: fraction of predictions within 1 step of true distance
- **metric_for_best_model:** MAE (lower is better)

### Inference

`score_candidates(model, tokenizer, current, target, candidates)` — predicts `distance(candidate, target)` for each candidate. Returns list sorted ascending (lower = closer = better).

Path generation: beam search picks neighbors with lowest predicted distance at each step. Falls back to BFS if beam search fails.

**Deployed HTTP API:** a FastAPI + Docker service in **`word-ladder-api/`** exposes the same neighbor scoring as `POST /predict` (modes `en_4`, `en_5`, `hr_4`, `hr_5`). **Public base URL (example):** `https://ldoric-word-ladder-api.hf.space` — use the **`.hf.space`** host for HTTP clients, not `huggingface.co/spaces/.../path` (see `docs/context.md` § BERT hint API). Full deploy notes: `word-ladder-api/README.md`. **Smoke test** all four Hub-backed models: `notebooks/15_model_api_test.ipynb`.

### Output

- **Trained model (English 5-letter):** `models/bert_wordladder_5letter/` (model + tokenizer)
- **Trained model (English 4-letter):** `models/bert_wordladder_4letter/` (model + tokenizer)
- **Trained model (Croatian 5-letter):** `models/bert_wordladder_croatian5/` (model + tokenizer)
- **Checkpoints (English 5-letter):** `outputs/bert_wordladder/` (per-epoch)
- **Checkpoints (English 4-letter):** `outputs/bert_wordladder_4letter/` (per-epoch)
- **Checkpoints (Croatian 5-letter):** `outputs/bert_wordladder_croatian5/` (per-epoch)
- **Trained model (Croatian 4-letter):** `models/bert_wordladder_croatian4/` (model + tokenizer)
- **Checkpoints (Croatian 4-letter):** `outputs/bert_wordladder_croatian4/` (per-epoch)

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
1. Run `notebooks/04_english_datasets_testing.ipynb` to produce English island files (if not present)
2. Run `notebooks/03_croatian_datasets_testing.ipynb` to produce Croatian island files (if not present)
3. Run `notebooks/05_english_5_letter_training.ipynb` to generate English 5-letter CSVs
4. Run `notebooks/07_english_4_letter_training.ipynb` to generate English 4-letter CSVs
5. Run `notebooks/09_croatian_5_letter_training.ipynb` to generate Croatian 5-letter CSVs
6. Run `notebooks/11_croatian_4_letter_training.ipynb` to generate Croatian 4-letter CSVs

### Step 2: Fine-tune
1. Run the corresponding fine-tune notebook:
   - `notebooks/06_bert_wordladder_finetune.ipynb` (English 5-letter)
   - `notebooks/08_bert_wordladder_4letter_finetune.ipynb` (English 4-letter)
   - `notebooks/10_bert_wordladder_croatian5_finetune.ipynb` (Croatian 5-letter)
   - `notebooks/12_bert_wordladder_croatian4_finetune.ipynb` (Croatian 4-letter)
2. Run the "Ensure accelerate" cell first if you get `ImportError` about accelerate
3. Let training complete (~5–10 min on GPU, longer on CPU)

### Step 3: Use the model
- English 5-letter model: `models/bert_wordladder_5letter/`
- English 4-letter model: `models/bert_wordladder_4letter/`
- Croatian 5-letter model: `models/bert_wordladder_croatian5/`
- Croatian 4-letter model: `models/bert_wordladder_croatian4/`
- Load with `AutoModelForSequenceClassification.from_pretrained("models/bert_wordladder_...")`
- Use `score_candidates()` for inference

---

## 5. Hardware Notes

### CPU (laptop)
- **Time:** Several hours for 5 epochs (80k examples, batch 32)
- **Note:** `pin_memory` warning is harmless on CPU
- **Device:** AMD Radeon laptops do not support PyTorch GPU (NVIDIA CUDA only)

### GPU (NVIDIA)
- **Time:** ~10–20 minutes for 5 epochs
- Install CUDA-enabled PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cu118`
- Trainer uses GPU automatically if available

### Google Colab
1. Upload/clone repo to Colab (or upload notebooks + data/islands/)
2. Runtime → Change runtime type → GPU (T4 free tier)
3. Run notebook 05 to generate CSVs (needs data/islands/)
4. Run notebook 06 to fine-tune (~10–20 min on GPU)
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

### Phase 2 runs (distance regression)

| Run | Graph | Model | Examples | Epochs | Val MAE | Test MAE | Test RMSE | Within ±1 | Notes |
|-----|-------|-------|----------|--------|---------|----------|-----------|-----------|-------|
| 5 (Colab GPU) | EN 5-letter | bert-base-uncased | 80k | 5 | 0.945 | 0.952 | 1.320 | 63.3% | first distance regression run |
| 6 (Colab T4) | EN 5-letter | bert-base-uncased | 250k | 5 | 0.781 | 0.783 | 1.096 | 72.5% | 3× more data |
| 7 (Colab L4) | EN 5-letter | bert-base-uncased | 600k | 6 | **0.588** | **0.587** | **0.792** | **83.6%** | weighted sampling + cosine — **best EN 5-letter** |
| 8 (Colab GPU) | EN 4-letter | bert-base-uncased | 600k | 6 | **0.311** | **0.308** | **0.430** | **97.3%** | 4-letter graph, same recipe — **best overall** |
| 9 (Colab GPU) | HR 5-letter | bert-base-uncased | 600k | 6 | 0.451 | 0.450 | 0.601 | 91.0% | Croatian 5-letter graph |
| 10 (Colab GPU) | **HR 4-letter** | bert-base-uncased | 600k | 6 | **0.229** | **0.230** | **0.327** | **99.1%** | Croatian 4-letter — **best overall** |

### Run 5: Distance regression (2025-03-22) — Colab T4

| Epoch | Train Loss | Val Loss | Val MAE | Val RMSE |
|-------|------------|----------|---------|----------|
| 1 | 3.80 | 2.97 | 1.281 | 1.724 |
| 2 | 2.47 | 2.35 | 1.125 | 1.533 |
| 3 | 1.79 | 1.85 | 0.999 | 1.361 |
| 4 | 1.40 | 1.76 | 0.961 | 1.326 |
| 5 | 1.16 | 1.73 | 0.945 | 1.315 |

- **Training time:** ~2249 s (~37 min) on Colab T4
- **Validation:** MAE 0.945, RMSE 1.315, within ±1 step: 64.9%
- **Test:** MAE 0.952, RMSE 1.320, within ±1 step: 63.3%

**Observations:**
- Train loss still decreasing steadily (3.80 → 1.16); val loss flattening (1.85 → 1.73 over last 3 epochs). Model is still learning but approaching diminishing returns.
- MAE < 1.0 means the model predicts distance within 1 step on average. For a graph with distances ranging 1–10+, this is meaningful learning.
- ~64% of predictions fall within ±1 step of the true distance.
- The missing/unexpected LayerNorm keys are the same harmless naming mismatch seen in all runs (beta/gamma vs weight/bias).

**What the numbers mean for path generation:**
- At each step, the model ranks neighbors by predicted distance to the target. What matters is not the absolute MAE but whether the model correctly identifies which neighbor is *closer*.
- A neighbor on a shortest path has true distance d−1; a wrong neighbor typically has distance d+1 (difference of 2). With MAE ~0.95, the model should often get this ranking right.
- Beam search (width 3) further mitigates occasional ranking errors by exploring multiple candidates.
- **The real test is path generation** — try the demo cells in notebook 06 to see if the model can guide paths without BFS fallback.

### Run 5: Path generation results

**Beam size 3:**

| Start → Target | Steps | Method | BFS optimal | Notes |
|----------------|-------|--------|-------------|-------|
| miles → tanks | 13 | beam | 4 | valid path but very suboptimal |
| light → right | 1 | beam | 1 | trivial 1-step |
| crane → flame | 6 | beam | ~5–6 | close to optimal |
| black → white | 10 | beam | ~7 | reasonable |

**Beam size 1 (greedy):**

| Start → Target | Steps | Method | Notes |
|----------------|-------|--------|-------|
| miles → tanks | 4 | bfs | greedy failed, BFS fallback |
| light → right | 1 | beam | trivial |
| crane → flame | 11 | beam | found path but very long |
| black → white | 7 | bfs | greedy failed, BFS fallback |

**Key result (beam search):** Beam 3 finds all 4 paths without BFS fallback — a complete reversal from Phase 1. But paths are suboptimal (e.g. 13 steps vs BFS-optimal 4) because beam search advances all paths in lockstep and doesn't penalize path length.

### Run 5: A\* search results

Replaced beam search with proper A\* (priority queue, `f = g + h` where g = steps so far, h = predicted distance). Batched inference for speed.

| Start → Target | Steps | Method | BFS optimal | Notes |
|----------------|-------|--------|-------------|-------|
| miles → tanks | 4 | astar | 4 | **optimal** — was 13 with beam |
| light → right | 1 | astar | 1 | trivial |
| crane → flame | 4 | astar | ~4 | **optimal** — was 6 with beam |
| black → white | 7 | bfs | ~7 | A\* exhausted 300 expansions, BFS fallback |

**Key result:** A\* finds **optimal or near-optimal paths** (3/4 via A\*, all at BFS-optimal length). This confirms the distance regression model works as a valid A\* heuristic. black→white still falls back to BFS — likely needs a better-trained heuristic (more data / lower MAE).

**Speed note:** Inference on laptop CPU is slow (~10 min for 4 paths). Run inference on Colab GPU for practical evaluation.

### Run 6: 250k examples (2025-03-22) — Colab T4

Increased data from 80k to 250k (2000 BFS sources). Same model and hyperparameters.

| Epoch | Train Loss | Val Loss | Val MAE | Val RMSE |
|-------|------------|----------|---------|----------|
| 1 | 2.15 | 1.96 | 1.025 | 1.399 |
| 2 | 1.78 | 1.53 | 0.893 | 1.237 |
| 3 | 1.31 | 1.27 | 0.825 | 1.125 |
| 4 | 1.08 | 1.20 | 0.801 | 1.095 |
| 5 | 0.94 | 1.16 | 0.781 | 1.076 |

- **Training time:** ~7570 s (~126 min) on Colab T4
- **Validation:** MAE 0.781, RMSE 1.076, within ±1 step: 72.5%
- **Test:** MAE 0.783, RMSE 1.096, within ±1 step: 72.5%

**Improvement over Run 5 (80k → 250k):**

| Metric | Run 5 (80k) | Run 6 (250k) | Improvement |
|--------|-------------|--------------|-------------|
| Test MAE | 0.952 | 0.783 | −0.169 (18% better) |
| Test RMSE | 1.320 | 1.096 | −0.224 (17% better) |
| Within ±1 | 63.3% | 72.5% | +9.2 pp |

Train loss still decreasing (0.94 at epoch 5); val loss flattening (1.27 → 1.20 → 1.16). More epochs could help slightly but diminishing returns.

### Run 6: A\* path generation (Colab GPU, 18 seconds for 4 paths)

| Start → Target | Steps | Method | BFS optimal | Notes |
|----------------|-------|--------|-------------|-------|
| miles → tanks | 4 | astar | 4 | optimal |
| light → right | 1 | astar | 1 | trivial |
| crane → flame | 4 | astar | ~4 | optimal |
| black → white | 7 | bfs | ~7 | A\* exhausted 300 expansions, BFS fallback |

Same A\* success rate as Run 5 (3/4), but with better MAE the heuristic is more accurate overall. black→white remains hard — the model's heuristic misleads A\* in that region of the graph. On Colab GPU: **18 seconds for all 4 paths** (vs 10 min on CPU).

### Run 6: Batch evaluation — 200 random pairs (notebook 06, cell 24)

Random pairs from `english_5_strict_largest_island.txt` with **BFS distance 3–10** (seed 42). Same Run 6 model, A\* + BFS fallback.

| Metric | Value |
|--------|-------|
| **A\* without fallback** | 140/200 (**70.0%**) |
| **BFS fallback** | 60/200 (**30.0%**) |
| **Path length = BFS optimal** | 178/200 (**89.0%**) |
| **Avg path length (A\* successes only)** | 6.76 vs optimal 6.59 (**ratio 1.03**) |
| **Wall time** | ~19 055 s (~5.3 h), **~95 s / pair** |

**Breakdown by true BFS distance:**

| Dist | Count | A\* OK | Optimal | Avg A\* len |
|------|-------|--------|---------|-------------|
| 3 | 4 | 4 | 4 | 3.00 |
| 4 | 13 | 13 | 12 | 4.08 |
| 5 | 24 | 23 | 22 | 5.09 |
| 6 | 41 | 33 | 34 | 6.24 |
| 7 | 37 | 23 | 33 | 7.22 |
| 8 | 31 | 18 | 28 | 8.17 |
| 9 | 33 | 19 | 30 | 9.16 |
| 10 | 17 | 7 | 15 | 10.29 |

**Interpretation (thesis-ready):**

- **89% shortest-length paths:** Even when A\* times out, BFS returns an optimal path — so the *solver* almost always returns a shortest ladder. The neural part is the **heuristic**; the pipeline is still **sound and rule-adherent** (every step is a valid one-letter change in the graph).
- **70% “pure” A\*:** Longer true distances (8–10) drive most fallbacks — the learned heuristic wanders before hitting the expansion limit. Short and medium ladders (3–6) are handled by A\* most of the time.
- **Ratio 1.03 on A\* successes:** When A\* reaches the target without fallback, average length is only ~3% above the true shortest — strong ranking / guidance, not random walk.
- **Latency:** ~95 s/pair suggests evaluation ran on **CPU** or without heavy batching across pairs; the 4-path Colab GPU demo was ~18 s total. For large studies, run the eval cell on **GPU** or lower `N_PAIRS` / raise `max_expansions` only where needed.

### Run 7: 600k + weighted sampling (2025-03-23) — Colab **L4**

Notebook 05: **600k** rows, **4000** BFS sources, **weighted** distance sampling. Notebook 06: **6 epochs**, cosine LR, **warmup_ratio=0.06**.

| Epoch | Train Loss | Val Loss | Val MAE | Val RMSE |
|-------|------------|----------|---------|----------|
| 1 | 1.85 | 1.59 | 0.935 | 1.262 |
| 2 | 1.13 | 1.08 | 0.774 | 1.040 |
| 3 | 0.89 | 0.81 | 0.671 | 0.900 |
| 4 | 0.66 | 0.72 | 0.629 | 0.847 |
| 5 | 0.51 | 0.65 | 0.596 | 0.808 |
| 6 | 0.47 | 0.64 | 0.588 | 0.799 |

- **Training time:** ~9715 s (~**2.7 h**) on Colab L4 (`train_samples_per_second` ~333.5)
- **Manual eval (notebook):** Val MAE **0.588**, RMSE **0.799**; Test MAE **0.587**, RMSE **0.792**, within ±1 step **83.4%** / **83.6%**

**Improvement over Run 6 (250k → Run 7):**

| Metric | Run 6 | Run 7 | Δ |
|--------|-------|-------|---|
| Test MAE | 0.783 | 0.587 | −0.196 (~25% relative) |
| Test RMSE | 1.096 | 0.792 | −0.304 |
| Within ±1 | 72.5% | 83.6% | +11.1 pp |

### Run 7: Batch evaluation — 200 pairs (GPU, seed 42)

| Metric | Run 6 (CPU) | Run 7 (L4 GPU) |
|--------|-------------|----------------|
| **A\* without fallback** | 70.0% | **94.0%** |
| **BFS fallback** | 30.0% | **6.0%** |
| **Path length = BFS optimal** | 89.0% | **95.0%** |
| **Avg A\* length ratio** | 1.03 | **1.01** |
| **Wall time (200 pairs)** | ~5.3 h | **~238 s** (~1.19 s/pair) |

**Breakdown by true BFS distance (Run 7):**

| Dist | Count | A\* OK | Optimal | Avg A\* len |
|------|-------|--------|---------|-------------|
| 3 | 4 | 4 | 4 | 3.00 |
| 4 | 13 | 13 | 13 | 4.00 |
| 5 | 24 | 24 | 24 | 5.00 |
| 6 | 41 | 41 | 39 | 6.05 |
| 7 | 37 | 36 | 32 | 7.14 |
| 8 | 31 | 27 | 31 | 8.00 |
| 9 | 33 | 31 | 31 | 9.10 |
| 10 | 17 | 12 | 16 | 10.08 |

### Run 7: Canonical path demos (all **astar**, 4 steps where BFS-optimal is 4)

| Start → Target | Steps | Method |
|----------------|-------|--------|
| miles → tanks | 4 | astar |
| light → right | 1 | astar |
| crane → flame | 4 | astar |
| black → white | 7 | astar |

**black → white** no longer requires BFS fallback (contrast Run 6).

**Thesis-ready summary:** Run 7 shows that **scaling data + emphasizing short graph distances + cosine schedule** sharply improves both **regression quality** (MAE ~0.59) and **A\* reliability** (94% pure A\*, 95% shortest-length paths on the 200-pair benchmark).

---

## English 4-Letter Distance Regression

### Overview

Same pipeline as 5-letter (Phase 2), but on the **4-letter English** word-ladder graph. Notebooks 07 (data gen) + 08 (fine-tune).

- **Graph:** 5,643 nodes (largest connected component from `english_4_largest_island.txt`)
- **Playable:** 3,155 strict words (`english_4_strict_largest_island.txt`)
- **Data:** 600k examples, 4,000 BFS sources, weighted sampling (same recipe as 5-letter Run 7)
- **Distance range:** 1–15, mean 5.02

### Run 8: 4-letter, 600k + weighted (2025-04-02) — Colab GPU

| Epoch | Train Loss | Val Loss | Val MAE | Val RMSE |
|-------|------------|----------|---------|----------|
| 1 | 0.530 | 0.461 | 0.514 | 0.679 |
| 2 | 0.319 | 0.293 | 0.407 | 0.541 |
| 3 | 0.230 | 0.235 | 0.360 | 0.484 |
| 4 | 0.180 | 0.206 | 0.333 | 0.454 |
| 5 | 0.151 | 0.183 | 0.313 | 0.428 |
| 6 | 0.128 | 0.182 | 0.311 | 0.426 |

- **Training time:** ~9743 s (~**2.7 h**) (`train_samples_per_second` ~332.6)
- **Validation:** MAE **0.3107**, RMSE **0.4262**, within ±1 step **97.12%**
- **Test:** MAE **0.3082**, RMSE **0.4304**, within ±1 step **97.26%**

**Comparison with 5-letter Run 7:**

| Metric | 5-letter Run 7 | 4-letter Run 8 | Δ |
|--------|----------------|----------------|---|
| Test MAE | 0.587 | **0.308** | −0.279 (~48% relative) |
| Test RMSE | 0.792 | **0.430** | −0.362 |
| Within ±1 | 83.6% | **97.3%** | +13.7 pp |

The 4-letter model is substantially better — smaller, denser graph with higher BFS-source coverage (71% of vocab vs 40%).

### Run 8: Canonical path demos (all **astar**)

| Start → Target | Steps | Method |
|----------------|-------|--------|
| cold → warm | 4 | astar |
| love → hate | 3 | astar |
| dark → pale | 3 | astar |
| head → tail | 4 | astar |

All 4 demo paths found by A\* without BFS fallback.

### Run 8: Batch evaluation — 200 pairs (GPU, seed 42)

| Metric | 5-letter Run 7 | 4-letter Run 8 |
|--------|----------------|----------------|
| **A\* without fallback** | 94.0% | **97.0%** |
| **BFS fallback** | 6.0% | **3.0%** |
| **Path length = BFS optimal** | 95.0% | **100.0%** |
| **Avg A\* length ratio** | 1.01 | **1.00** |
| **Wall time (200 pairs)** | ~238 s | ~4464 s (~22 s/pair) |

**Breakdown by true BFS distance (Run 8):**

| Dist | Count | A\* OK | Optimal | Avg A\* len |
|------|-------|--------|---------|-------------|
| 3 | 25 | 25 | 25 | 3.00 |
| 4 | 57 | 57 | 57 | 4.00 |
| 5 | 53 | 51 | 53 | 5.00 |
| 6 | 51 | 47 | 51 | 6.00 |
| 7 | 12 | 12 | 12 | 7.00 |
| 8 | 2 | 2 | 2 | 8.00 |

**Key result:** **100% of paths are BFS-optimal length** — even the 6 BFS-fallback cases return shortest paths. A\* alone solves 97% without fallback. Every A\* success produces a path at exactly the optimal length (ratio 1.00). The 4-letter graph's denser connectivity and the model's low MAE (0.308) make the heuristic near-perfect for neighbor ranking.

---

## Croatian 5-Letter Distance Regression

### Overview

Same pipeline as English, but on the **Croatian 5-letter** word-ladder graph. Notebooks 09 (data gen) + 10 (fine-tune).

- **Graph:** 11,052 nodes (largest connected component from `croatian_5_largest_island.txt`)
- **Playable:** 1,448 strict words (`croatian_5_strict_largest_island.txt`)
- **Alphabet:** `abcdefghijklmnopqrstuvwxyzčćđšž` (31 letters). Digraphs `lj`, `nj`, `dž` are treated as two separate characters (consistent with how all word files store 5-character words).
- **Data:** 600k examples, 4,000 BFS sources, weighted sampling (same recipe as English)
- **Tokenizer note:** `bert-base-uncased` was not pretrained on Croatian. Croatian characters (č, ć, đ, š, ž) are split into subword/character tokens. The model learns distance patterns from these during fine-tuning.

### Run 9: Croatian 5-letter, 600k + weighted (2025-04-02) — Colab GPU

| Epoch | Train Loss | Val Loss | Val MAE | Val RMSE |
|-------|------------|----------|---------|----------|
| 1 | 1.037 | 0.858 | 0.690 | 0.926 |
| 2 | 0.620 | 0.604 | 0.589 | 0.777 |
| 3 | 0.464 | 0.475 | 0.522 | 0.689 |
| 4 | 0.354 | 0.389 | 0.466 | 0.624 |
| 5 | 0.295 | 0.379 | 0.462 | 0.616 |
| 6 | 0.264 | 0.363 | 0.451 | 0.603 |

- **Training time:** ~9540 s (~**2.7 h**) (`train_samples_per_second` ~339.6)
- **Validation:** MAE **0.4512**, RMSE **0.6028**, within ±1 step **91.08%**
- **Test:** MAE **0.4504**, RMSE **0.6006**, within ±1 step **90.96%**

**Cross-language comparison:**

| Metric | EN 5-letter (Run 7) | HR 5-letter (Run 9) | EN 4-letter (Run 8) |
|--------|---------------------|---------------------|---------------------|
| Test MAE | 0.587 | **0.450** | 0.308 |
| Test RMSE | 0.792 | **0.601** | 0.430 |
| Within ±1 | 83.6% | **91.0%** | 97.3% |

Croatian outperforms English 5-letter despite `bert-base-uncased` having no Croatian pretraining — likely because the Croatian graph has more nodes (11,052 vs 9,902) with similar avg degree, giving richer training signal.

### Run 9: Canonical path demos

| Start → Target | Steps | Method | Notes |
|----------------|-------|--------|-------|
| banka → arena | 6 | astar | banka→danka→daska→drska→dreka→drena→arena |
| bajka → balet | 11 | bfs | A\* exhausted expansions, BFS fallback (long path) |
| autor → badem | 6 | astar | autor→autom→butom→batom→barom→barem→badem |
| kutak → blato | — | — | no path found (different component or word not in vocab) |

### Run 9: Batch evaluation — 200 pairs (GPU, seed 42)

| Metric | EN 5-letter Run 7 | HR 5-letter Run 9 | EN 4-letter Run 8 |
|--------|-------------------|-------------------|-------------------|
| **A\* without fallback** | 94.0% | **98.5%** | 97.0% |
| **BFS fallback** | 6.0% | **1.5%** | 3.0% |
| **Path length = BFS optimal** | 95.0% | **98.0%** | 100.0% |
| **Avg A\* length ratio** | 1.01 | **1.00** | 1.00 |
| **Wall time (200 pairs)** | ~238 s | ~1774 s (~8.9 s/pair) | ~4464 s |

**Breakdown by true BFS distance (Run 9):**

| Dist | Count | A\* OK | Optimal | Avg A\* len |
|------|-------|--------|---------|-------------|
| 3 | 3 | 3 | 3 | 3.00 |
| 4 | 8 | 8 | 8 | 4.00 |
| 5 | 14 | 14 | 14 | 5.00 |
| 6 | 22 | 22 | 22 | 6.00 |
| 7 | 34 | 34 | 34 | 7.00 |
| 8 | 38 | 37 | 37 | 8.03 |
| 9 | 42 | 41 | 40 | 9.05 |
| 10 | 39 | 38 | 38 | 10.05 |

**Key result:** The Croatian model achieves **98.5% pure A\*** and **98.0% optimal paths** — the best A\* success rate across all models. Only 3 out of 200 pairs needed BFS fallback. Even at distances 8–10, A\* almost always finds the target within 300 expansions. The length ratio of 1.00 means A\* paths are essentially always shortest-length.

---

## Croatian 4-Letter Distance Regression

### Overview

Same pipeline, now on the **Croatian 4-letter** word-ladder graph. Notebooks 11 (data gen) + 12 (fine-tune).

- **Graph:** 3,863 nodes (largest connected component from `croatian_4_largest_island.txt`)
- **Playable:** 941 strict words (`croatian_4_strict_largest_island.txt`)
- **Alphabet:** `abcdefghijklmnopqrstuvwxyzčćđšž` (31 letters)
- **Data:** 600k examples, 3,500 BFS sources (~91% vocab coverage — highest of all models), weighted sampling
- **Avg degree:** ~13.8 (densest graph of the four)

### Run 10: Croatian 4-letter, 600k + weighted (2025-04-02) — Colab GPU

| Epoch | Train Loss | Val Loss | Val MAE | Val RMSE |
|-------|------------|----------|---------|----------|
| 1 | 0.330 | 0.308 | 0.417 | 0.555 |
| 2 | 0.202 | 0.184 | 0.317 | 0.429 |
| 3 | 0.154 | 0.148 | 0.285 | 0.385 |
| 4 | 0.124 | 0.124 | 0.250 | 0.352 |
| 5 | 0.104 | 0.110 | 0.232 | 0.331 |
| 6 | 0.096 | 0.108 | 0.229 | 0.329 |

- **Training time:** ~9672 s (~**2.7 h**) (`train_samples_per_second` ~335.0)
- **Validation:** MAE **0.2288**, RMSE **0.3287**, within ±1 step **99.06%**
- **Test:** MAE **0.2304**, RMSE **0.3273**, within ±1 step **99.13%**

**Cross-model comparison (all runs):**

| Metric | EN 5L (R7) | EN 4L (R8) | HR 5L (R9) | HR 4L (R10) |
|--------|------------|------------|------------|-------------|
| Test MAE | 0.587 | 0.308 | 0.450 | **0.230** |
| Test RMSE | 0.792 | 0.430 | 0.601 | **0.327** |
| Within ±1 | 83.6% | 97.3% | 91.0% | **99.1%** |

Croatian 4-letter achieves the lowest MAE and highest within-±1 across all models. The combination of dense graph (avg degree ~13.8), small vocab (3,863), and high BFS-source coverage (~91%) makes the distance function nearly perfectly learnable.

### Run 10: Canonical path demos (all **astar**)

| Start → Target | Steps | Method | Path |
|----------------|-------|--------|------|
| baka → ruka | 2 | astar | baka→raka→ruka |
| kula → most | 4 | astar | kula→kola→kosa→kost→most |
| grad → selo | 7 | astar | grad→graf→grof→groa→grla→gela→sela→selo |
| more → zima | 5 | astar | more→mome→tome→time→tima→zima |

All 4 demo paths found by A\* without BFS fallback.

### Run 10: Batch evaluation — 200 pairs (GPU, seed 42)

| Metric | EN 5L (R7) | EN 4L (R8) | HR 5L (R9) | HR 4L (R10) |
|--------|------------|------------|------------|-------------|
| **A\* without fallback** | 94.0% | 97.0% | 98.5% | **98.0%** |
| **BFS fallback** | 6.0% | 3.0% | 1.5% | **2.0%** |
| **Path length = BFS optimal** | 95.0% | 100.0% | 98.0% | **100.0%** |
| **Avg A\* length ratio** | 1.01 | 1.00 | 1.00 | **1.00** |
| **Wall time (200 pairs)** | ~238 s | ~4464 s | ~1774 s | ~2553 s (~12.8 s/pair) |

**Breakdown by true BFS distance (Run 10):**

| Dist | Count | A\* OK | Optimal | Avg A\* len |
|------|-------|--------|---------|-------------|
| 3 | 22 | 22 | 22 | 3.00 |
| 4 | 33 | 33 | 33 | 4.00 |
| 5 | 37 | 37 | 37 | 5.00 |
| 6 | 18 | 18 | 18 | 6.00 |
| 7 | 24 | 24 | 24 | 7.00 |
| 8 | 33 | 31 | 33 | 8.00 |
| 9 | 13 | 13 | 13 | 9.00 |
| 10 | 20 | 18 | 20 | 10.00 |

**Key result:** **100% optimal paths** (tied with English 4-letter) and **98% pure A\***. Even the 4 BFS-fallback cases return shortest paths. Perfect A\* length at every distance bucket except 8 and 10 (where 2 pairs each exhausted 300 expansions). The model's MAE of 0.23 means predictions are off by less than a quarter of a step on average — effectively perfect ranking for A\* neighbor selection.

---

## 8. Next steps

### Evaluation

Run 7 batch eval on **GPU** is logged above. Optional: **500 pairs** for tighter CIs; same seed for comparability.

### Further training improvements (optional)

| Change | Why |
|--------|-----|
| **Continuation fine-tune** | Load Run 7 checkpoint, 2–3 epochs, LR ~1e-5, if val still improving. |
| **DeBERTa-v3-base** | Only if chasing last points; Run 7 is already strong on BERT-base. |

---

## 9. Changelog

- **2025-03-20:** Created training pipeline (05 data gen, 06 BERT finetune)
- **2025-03-20:** First CPU training run completed — val 77.45%, test 79.51%
- **2025-03-20:** Fixed evaluate cell (manual eval to avoid callback error)
- **2025-03-20:** Added improvement ideas
- **2025-03-20:** Raised cap to 50k, switched to roberta-base, added .gitignore for models/outputs/training CSVs
- **2025-03-20:** RoBERTa Colab run — val 75.80%, test 74.13% (worse than BERT baseline). Added colab-setup.md
- **2025-03-20:** BERT Colab run on same dataset — val 82.84%, test 81.75% (best classification run)
- **2025-03-22:** Path generation testing: model fails for multi-step paths. BFS fallback added. Analysis added.
- **2025-03-22:** Beam search implemented. Multi-positive and harder negatives tested — did not help (Run 4).
- **2025-03-22:** **Switched to distance regression.** Rewrote notebook 05 (BFS distance pairs, 80k examples) and notebook 06 (regression head, MSE loss, MAE metrics). Updated play_wordladder.py for distance-based scoring. Phase 1 results archived.
- **2025-03-22:** Run 5 (Colab T4): distance regression — val MAE 0.945, test MAE 0.952, within ±1 step 63.3%. Model predicts distance within ~1 step on average.
- **2025-03-22:** Path generation test: beam 3 finds all 4 test paths without BFS fallback (first time ever!). Paths are valid but longer than optimal (e.g. 13 steps vs BFS-optimal 4).
- **2025-03-22:** Replaced beam search with proper A\* (priority queue + batched inference). Results: 3/4 paths found by A\* at BFS-optimal length (miles→tanks 4 steps, crane→flame 4 steps). black→white still needs BFS fallback.
- **2025-03-22:** Run 6 (250k examples, Colab T4): MAE 0.783 (down from 0.952), within ±1 step 72.5%. A\* still 3/4, 18 seconds on GPU.
- **2025-03-23:** Batch eval (200 pairs, dist 3–10): 70% pure A\*, 30% BFS fallback, **89% optimal path length**, A\*-only avg length ratio 1.03 vs BFS optimal. ~5.3 h total (~95 s/pair — likely CPU eval).
- **2025-03-23:** Run 7 prep: notebook 05 → **600k** examples, **4000** BFS sources, **weighted** distance sampling; notebook 06 → **6 epochs**, **cosine** LR + **warmup_ratio=0.06**.
- **2025-03-23:** **Run 7 complete** (Colab L4): test MAE **0.587**, within ±1 **83.6%**, train ~**2.7 h**. Batch eval: **94%** pure A\*, **6%** BFS, **95%** optimal length, **~4 min** for 200 pairs on GPU. All four demo paths including black→white via **astar**.
- **2025-04-02:** Created notebooks 07 (4-letter data gen) and 08 (4-letter BERT finetune), mirroring 05/06 pipeline.
- **2025-04-02:** Notebook 07: 600k examples from 4-letter graph (5,643 nodes), 4,000 BFS sources, weighted sampling. Distance range 1–15, mean 5.02.
- **2025-04-02:** **Run 8 complete** (Colab GPU): 4-letter model — test MAE **0.308**, within ±1 **97.3%**, train ~**2.7 h**. All 4 demo paths via **astar**. ~48% lower MAE than 5-letter Run 7.
- **2025-04-02:** Run 8 batch eval (200 pairs, dist 3–10): **97%** pure A\*, **3%** BFS, **100% optimal path length**, ratio **1.00**. ~22 s/pair on GPU.
- **2025-04-02:** Created notebooks 09 (Croatian 5-letter data gen) and 10 (Croatian 5-letter BERT finetune).
- **2025-04-02:** **Run 9 complete** (Colab GPU): Croatian 5-letter model — test MAE **0.450**, within ±1 **91.0%**, train ~**2.7 h**. Batch eval: **98.5%** pure A\*, **1.5%** BFS, **98.0%** optimal length, ratio **1.00**. Best A\* success rate across all models.
- **2025-04-02:** Created notebooks 11 (Croatian 4-letter data gen) and 12 (Croatian 4-letter BERT finetune).
- **2025-04-02:** **Run 10 complete** (Colab GPU): Croatian 4-letter model — test MAE **0.230**, within ±1 **99.1%**, train ~**2.7 h**. Batch eval: **98%** pure A\*, **2%** BFS, **100% optimal path length**, ratio **1.00**. Best MAE across all models. All 4 demo paths via **astar**.
