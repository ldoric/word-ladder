# Training Log

Everything we did for Word Ladder model training: data generation, BERT fine-tuning, and results.

---

## Overview

**Goal:** Fine-tune BERT to predict the **BFS shortest-path distance** between two words in the word-ladder graph (5-letter English). Use this as an A\* heuristic for path generation.

**Task:** Regression (MSE loss)  
- **Input:** `text_a` = word\_a, `text_b` = word\_b (BERT sentence pair)  
- **Label:** shortest-path distance (integer, e.g. 3)  

**Pipeline:**  
1. Notebook 05: generate distance-regression training data (CSVs)  
2. Notebook 06: fine-tune BERT, evaluate, save model  

**Colab:** Run notebooks 05 and 06 on Google Colab (Runtime → GPU) for faster training. `models/`, `outputs/`, `data/training/*.csv` are gitignored — regenerate on Colab or upload.  

**Run 7+ notebook defaults (scale-up):**

| Setting | Notebook 05 | Notebook 06 |
|--------|-------------|-------------|
| Data size | **600k** examples | — |
| BFS sources | **4000** | — |
| Sampling | **Weighted** (higher weight for graph distance ≤3, then ≤6, then ≤9) | — |
| Training | — | **6 epochs**, **cosine** LR schedule, **warmup_ratio=0.06** |
| Logging | — | `logging_steps=100` |

**Do you need to rerun 05 and 06?** **Yes.** After these changes, run **05 first** (new CSVs), then **06** (train on those CSVs). Old **250k** CSVs + old checkpoints are **not** equivalent to Run 7 data/schedule.

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

### Output

- **Trained model:** `models/bert_wordladder_5letter/` (model + tokenizer)
- **Checkpoints:** `outputs/bert_wordladder/` (per-epoch)

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
3. Let training complete (~5–10 min on GPU, longer on CPU)

### Step 3: Use the model
- Model saved to `models/bert_wordladder_5letter/`
- Load with `AutoModelForSequenceClassification.from_pretrained("models/bert_wordladder_5letter")`
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

| Run | Model | Examples | Epochs | Val MAE | Test MAE | Test RMSE | Within ±1 | Notes |
|-----|-------|----------|--------|---------|----------|-----------|-----------|-------|
| 5 (Colab GPU) | bert-base-uncased | 80k | 5 | 0.945 | 0.952 | 1.320 | 63.3% | first distance regression run |
| 6 (Colab GPU) | bert-base-uncased | 250k | 5 | **0.781** | **0.783** | 1.096 | **72.5%** | 3× more data — best so far |

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

---

## 8. Next steps

### Evaluation

Batch eval (200 pairs) is logged above. Optional: repeat on GPU for faster runs, or extend to 500 pairs for tighter confidence intervals.

### Further training improvements (optional)

| Change | Why |
|--------|-----|
| **More epochs (8–10)** | Train loss still dropping at epoch 5. |
| **Lower learning rate (1e-5)** | Finer convergence in later epochs. |
| **Run 7: 600k + weighted sampling** | Implemented in notebook 05; retrain on Colab (see Overview table). |
| **DeBERTa-v3-base (optional)** | Try after Run 7 if you want a stronger encoder (VRAM permitting). |

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
- **2025-03-23:** Run 7 prep: notebook 05 → **600k** examples, **4000** BFS sources, **weighted** distance sampling; notebook 06 → **6 epochs**, **cosine** LR + **warmup_ratio=0.06**. **Rerun 05 then 06 on Colab** for Run 7 model.
