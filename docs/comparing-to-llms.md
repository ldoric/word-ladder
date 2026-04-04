# Comparing BERT + A\* to LLMs

Working notes for benchmark runs from `notebooks/13_llm_vs_bert_wordladder.ipynb` (same eval pairs for both systems).

---

## Run: English 5-letter — GPT `gpt-5.4-mini` vs BERT

**Setup (as run):**

- **Preset:** `english_5` (island graph + `bert_wordladder_5letter`)
- **Eval pairs:** 20 (named set e.g. `test5english` / `data/eval_sets/…`)
- **OpenAI model:** `gpt-5.4-mini` (sometimes written “5.4 nano”)
- **Hardware note:** BERT path timing below is **CPU** (~36 s/pair); GPU would be much faster.

### Your model (A\* + BFS fallback)

| Metric | Value |
|--------|-------|
| Wall time (20 pairs) | **724.6 s** (~**36.23 s** / pair) |
| A\* only | 18 / 20 |
| BFS fallback | 2 / 20 |
| **Valid ladder** (on graph, start→target) | **100%** |
| **Optimal length** (steps = sampled BFS distance) | **100%** |

Per-row behaviour: every run produced a valid path on the graph at optimal length (2× BFS fallback still optimal).

Sample BERT rows (`df_bert.head()`):

| | start | target | bfs_optimal | method | steps | valid | optimal_length |
|---|-------|--------|-------------|--------|-------|-------|----------------|
| 0 | worst | coils | 6 | astar | 6 | True | True |
| 1 | auger | husks | 6 | astar | 6 | True | True |
| 2 | decaf | clack | 10 | astar | 10 | True | True |
| 3 | sulky | capos | 8 | bfs | 8 | True | True |
| 4 | tufts | prier | 8 | astar | 8 | True | True |

### GPT (`gpt-5.4-mini`)

| Metric | Value |
|--------|-------|
| Wall time (20 pairs) | **~14.7 s** total |
| **Valid** (per notebook mode at time of run) | **10%** (2 / 20) |
| **Optimal length** (vs same BFS distance label) | **0%** |
| Mean steps when counted “success” | **5.00** (sparse valid rows) |

### Side-by-side summary

| Metric | BERT A\* + BFS | GPT `gpt-5.4-mini` |
|--------|----------------|----------------------|
| Valid / success | **100%** | **10%** |
| Optimal length (vs sampled opt) | **100%** | **0%** |
| Mean steps (when success) | **6.85** | 5.00 |
| Total wall seconds | 724.6 | ~14.7 |

### GPT failures observed (sample)

Typical failure modes: **invalid one-letter transitions** (model invents intermediate strings or skips letters), **no parse** (output not usable as JSON array), or **same word repeated** as a “step”.

| start | target | bfs_optimal | fail_reason (abridged) |
|-------|--------|-------------|-------------------------|
| worst | coils | 6 | not one-letter step: `coast` → `coats` |
| auger | husks | 6 | not one-letter step: `augus` → `hagus` |
| decaf | clack | 10 | not one-letter step: `decal` → `clack` |
| sulky | capos | 8 | not one-letter step: `cucky` → `cacks` |
| tufts | prier | 8 | not one-letter step: `prnes` → `prier` |
| bally | bahts | 4 | not one-letter step: `bally` → `bally` |
| chant | gangs | 9 | no parse |
| teddy | flier | 10 | not one-letter step: `teedy` → `teier` |
| zoned | suite | 8 | not one-letter step: `soned` → `sonie` |
| pouts | gaudy | 6 | not one-letter step: `pouts` → `pouts` |
| rents | tsars | 4 | not one-letter step: `tents` → `tarts` |
| drake | primp | 6 | not one-letter step: `grape` → `grime` |
| doper | fuses | 5 | not one-letter step: `coves` → `cures` |
| maser | civic | 6 | not one-letter step: `macer` → `macer` |
| cents | often | 7 | not one-letter step: `cooto` → `ootto` |

### Interpretation (short)

- **Rule adherence:** BERT + graph search respects one-letter moves and the lexicon by construction; the LLM must guess a chain without the full vocab and often violates edit distance or uses non-words.
- **Latency vs quality:** GPT is much faster per query but **unreliable** on this task at the tested settings; BERT is slower on CPU but **sound** for your game rules.
- **Fairness:** Compare at equal validation (graph vocab + one-letter checks) for thesis claims; relaxed LLM scoring is optional and labels should say it is not “in-game” valid.

---

## Run: English 5-letter — GPT `gpt-5.4` (full) vs BERT

**Setup:** Same as the mini run — `english_5`, **20 pairs** (`test5english`), same `df_bert` / `bert_time` from the earlier BERT benchmark. GPT model id: **`gpt-5.4`**. Validation mode unchanged (`GPT_REQUIRE_GRAPH_VOCAB` as in notebook for this run — same relaxed vs strict convention as mini).

### GPT (`gpt-5.4`)

| Metric | Value |
|--------|-------|
| Wall time (20 pairs) | **33.9 s** total (~**1.70 s** / pair) |
| **Valid / success** | **0%** (0 / 20) |
| **Optimal length** | **0%** |
| Mean steps (when success) | **n/a** (no successful rows; summary printed `nan`) |

### Side-by-side summary (same BERT numbers as above)

| Metric | BERT A\* + BFS | GPT `gpt-5.4` |
|--------|----------------|---------------|
| Valid / success | **100%** | **0%** |
| Optimal length (vs sampled opt) | **100%** | **0%** |
| Mean steps (when success) | **6.85** | n/a |
| Total wall seconds | 724.6 | 33.9 |

### GPT failures observed (sample)

| start | target | bfs_optimal | fail_reason (abridged) |
|-------|--------|-------------|-------------------------|
| worst | coils | 6 | not one-letter step: `worst` → `works` |
| auger | husks | 6 | not one-letter step: `auger` → `urger` |
| decaf | clack | 10 | not one-letter step: `decal` → `local` |
| sulky | capos | 8 | not one-letter step: `dally` → `dolls` |
| tufts | prier | 8 | not one-letter step: `tufts` → `turfs` |
| bally | bahts | 4 | not one-letter step: `balky` → `barks` |
| chant | gangs | 9 | not one-letter step: `clang` → `clang` |
| golds | slaws | 5 | not one-letter step: `holds` → `holds` |
| teddy | flier | 10 | no parse |
| zoned | suite | 8 | no parse |
| pouts | gaudy | 6 | not one-letter step: `gouty` → `gaudy` |
| rents | tsars | 4 | not one-letter step: `pants` → `pants` |
| drake | primp | 6 | not one-letter step: `brake` → `brine` |
| doper | fuses | 5 | not one-letter step: `fives` → `fuses` |
| maser | civic | 6 | not one-letter step: `manor` → `canon` |

### Mini vs full on this slice (same 20 pairs)

| Metric | `gpt-5.4-mini` | `gpt-5.4` |
|--------|----------------|-----------|
| Valid / success | 10% | **0%** |
| Total wall seconds | ~14.7 | 33.9 |

**Takeaway:** On this benchmark, **the larger model did not improve rule satisfaction**; it failed every ladder under the same checks. Plausible reasons: sampling/decoding differences, more “creative” chains that still break single-edit constraints, and **no access to your island lexicon** — capability in the abstract does not substitute for search over the real graph. Cost/latency went up (33.9 s vs ~14.7 s) without benefit here.

---

## Same 20 pairs — GPT `gpt-5.4` vs Gemini `gemini-2.5-flash` vs BERT

**Setup:** `english_5`, **20 pairs** (`test5english`), notebook 13. **GPT** = `gpt-5.4`, **Gemini** = `gemini-2.5-flash`. Same prompt builder (`build_gpt_chat_payload`) for both APIs. Validation flag as in notebook (`GPT_REQUIRE_GRAPH_VOCAB`); the printed summary **valid / success** follows that mode; extra Gemini diagnostics below include **strict island** validity.

### Summary (reproducibility row)

| Metric | BERT A\* + BFS | GPT `gpt-5.4` | Gemini `gemini-2.5-flash` |
|--------|----------------|---------------|---------------------------|
| Valid / success (per summary mode) | **100%** | **0%** | **5%** (1 / 20) |
| Optimal length (vs sampled BFS dist) | **100%** | **0%** | **5%** |
| Mean steps (when success) | **6.85** | n/a | **4.00** |
| Total wall seconds | **724.6** | 33.9 | 53.6 |

### Gemini — extra diagnostics (same run)

| Metric | Value |
|--------|-------|
| Mean wall time / pair | **~2.68 s** |
| **Valid on island graph** (`valid_on_graph`) | **0%** |
| Mean fraction of path words in graph (when any path returned) | **~0.34** |
| Optimal length among rows counted valid in summary mode | **100%** (single success was optimal vs `bfs_optimal`) |

So Gemini **beats GPT 5.4 on relaxed validity** on this slice (one success vs none) but still **never** produced a ladder that is fully valid **on your island graph** in this run. Latency is higher than GPT’s session total wall time (53.6 s vs 33.9 s for 20 pairs).

### Gemini failures (sample from run output)

| start | target | bfs_optimal | fail_reason (abridged) |
|-------|--------|-------------|-------------------------|
| drake | primp | 6 | no parse |
| doper | fuses | 5 | not one-letter step: `doper` → `poser` |
| maser | civic | 6 | no parse |
| cents | often | 7 | no parse |

(GPT failure list for this session matches the **`gpt-5.4`** table in the section above — 0% valid.)

### Interpretation (short)

- **Gemini 2.5 Flash** showed **occasional** relaxed-valid ladders where **GPT 5.4** had none, but **graph-strict** success remained **0%** for both.
- **BERT + search** remains the only approach with **100% / 100%** on this eval under your rules.

---

## English 4-letter — BERT vs GPT `gpt-5.4-mini` vs Gemini `gemini-2.5-flash`

**Setup:** `english_4`, **20 pairs** (`test4english` / `data/eval_sets/…`), notebook 13. **GPT** = `gpt-5.4-mini`, **Gemini** = `gemini-2.5-flash`. BERT timing below is **CPU** (~33 s/pair).

### BERT (A\* only; no BFS fallback this run)

| Metric | Value |
|--------|-------|
| Wall time (20 pairs) | **661.0 s** (~**33.05 s** / pair) |
| Method | **A\* 20 / 20** |
| **Valid ladder** | **100%** |
| **Optimal length** (vs sampled BFS distance) | **100%** |

`df_bert.head()`:

| | start | target | bfs_optimal | method | steps | valid | optimal_length |
|---|-------|--------|-------------|--------|-------|-------|----------------|
| 0 | subs | clue | 6 | astar | 6 | True | True |
| 1 | arms | wife | 5 | astar | 5 | True | True |
| 2 | hajj | gays | 4 | astar | 4 | True | True |
| 3 | flus | dali | 5 | astar | 5 | True | True |
| 4 | wets | chaw | 5 | astar | 5 | True | True |

### LLMs — summary row

| Metric | BERT A\* | GPT `gpt-5.4-mini` | Gemini `gemini-2.5-flash` |
|--------|----------|--------------------|---------------------------|
| Valid / success | **100%** | **5%** | **0%** |
| Optimal length (vs sampled opt) | **100%** | **0%** | **0%** |
| Mean steps (when success) | **5.35** | **2.00** | n/a |
| Total wall seconds | **661.0** | **15.1** | **56.2** |

### Gemini — diagnostics (same run)

| Metric | Value |
|--------|-------|
| Mean API latency / pair | **~2.81 s** |
| **Valid on island graph** | **0%** |
| Mean fraction of path words in graph | **~0.52** |

### GPT failures (sample)

| start | target | bfs_optimal | fail_reason (abridged) |
|-------|--------|-------------|-------------------------|
| subs | clue | 6 | not one-letter step: `subs` → `subs` |
| arms | wife | 5 | not one-letter step: `arms` → `arms` |
| hajj | gays | 4 | not one-letter step: `gajj` → `gays` |
| flus | dali | 5 | no parse |
| wets | chaw | 5 | not one-letter step: `wets` → `cats` |
| togs | whom | 5 | not one-letter step: `wogm` → `whom` |
| rape | byes | 5 | not one-letter step: `rype` → `ryes` |
| seam | mick | 5 | not one-letter step: `seam` → `seam` |
| care | flap | 6 | no parse |
| from | pics | 5 | not one-letter step: `from` → `firm` |
| shmo | asea | 7 | not one-letter step: `shmo` → `shoa` |
| ripe | ewer | 6 | not one-letter step: `ripe` → `wire` |
| vino | taco | 5 | not one-letter step: `vino` → `vino` |
| unco | rant | 7 | not one-letter step: `unco` → `runc` |
| meta | flee | 6 | not one-letter step: `mete` → `meet` |

### Gemini failures (sample)

| start | target | bfs_optimal | fail_reason (abridged) |
|-------|--------|-------------|-------------------------|
| shmo | asea | 7 | no parse |
| ripe | ewer | 6 | not one-letter step: `were` → `ewer` |
| vino | taco | 5 | not one-letter step: `vico` → `taco` |
| unco | rant | 7 | not one-letter step: `unto` → `ants` |

**Note:** On this 4-letter slice, **GPT-mini** had **one** relaxed-valid ladder (5% — hence mean steps 2.00); **Gemini** had **none** (0% valid). BERT remained **100% / 100%**.

---

## Croatian 5-letter — BERT vs GPT `gpt-5.4-mini` vs Gemini `gemini-2.5-flash`

**Setup:** `croatian_5`, **20 pairs** (`test5croatian`), notebook 13. **GPT** = `gpt-5.4-mini`, **Gemini** = `gemini-2.5-flash`. Prompt still includes the short **English** worked example for JSON shape; task lines use Croatian **start** / **target** (with letters č, ć, đ, š, ž as in the island files).

### BERT (A\* only)

| Metric | Value |
|--------|-------|
| Wall time (20 pairs) | **258.6 s** (~**12.93 s** / pair) |
| Method | **A\* 20 / 20** |
| **Valid ladder** | **100%** |
| **Optimal length** (vs sampled BFS distance) | **100%** |

`df_bert.head()`:

| | start | target | bfs_optimal | method | steps | valid | optimal_length |
|---|-------|--------|-------------|--------|-------|-------|----------------|
| 0 | zelen | genom | 7 | astar | 7 | True | True |
| 1 | lovit | lakši | 10 | astar | 10 | True | True |
| 2 | hitan | fešta | 9 | astar | 9 | True | True |
| 3 | šetaš | trčiš | 5 | astar | 5 | True | True |
| 4 | psiha | brava | 6 | astar | 6 | True | True |

### LLMs — summary row

| Metric | BERT A\* | GPT `gpt-5.4-mini` | Gemini `gemini-2.5-flash` |
|--------|----------|--------------------|---------------------------|
| Valid / success | **100%** | **5%** (1 / 20) | **5%** (1 / 20) |
| Optimal length (vs sampled opt) | **100%** | **0%** | **0%** |
| Mean steps (when success) | **7.70** | **4.00** | **4.00** |
| Total wall seconds | **258.6** | **14.7** | **54.6** |

### Gemini — diagnostics (same run)

| Metric | Value |
|--------|-------|
| Mean API latency / pair | **~2.73 s** |
| **Valid on island graph** | **0%** |
| Mean fraction of path words in graph | **~0.11** |
| Optimal length when rows counted “valid” in summary mode | **0%** (the single relaxed-valid ladder was **not** shortest vs `bfs_optimal`) |

### GPT failures (sample)

| start | target | bfs_optimal | fail_reason (abridged) |
|-------|--------|-------------|-------------------------|
| zelen | genom | 7 | not one-letter step: `zelen` → `genen` |
| lovit | lakši | 10 | not one-letter step: `lovit` → `lakit` |
| hitan | fešta | 9 | no parse |
| šetaš | trčiš | 5 | not one-letter step: `šetaš` → `šetri` |
| psiha | brava | 6 | not one-letter step: `psiha` → `prava` |
| bomba | ukini | 7 | not one-letter step: `bomba` → `ukina` |
| život | trčiš | 8 | not one-letter step: `život` → `trčot` |
| prođu | kvrga | 8 | not one-letter step: `prođu` → `proua` |
| mogla | bakar | 10 | not one-letter step: `mogla` → `mogar` |
| psuju | osoba | 10 | not one-letter step: `osuju` → `osoba` |
| fešta | držiš | 10 | not one-letter step: `dršta` → `držiš` |
| vidiš | melju | 7 | not one-letter step: `vidiš` → `vidju` |
| bujan | sedam | 6 | not one-letter step: `budan` → `sedan` |
| vučeš | voliš | 6 | not one-letter step: `vočiš` → `vočiš` |
| pekao | uroni | 8 | not one-letter step: `pekao` → `ukaoi` |

### Gemini failures (sample)

| start | target | bfs_optimal | fail_reason (abridged) |
|-------|--------|-------------|-------------------------|
| vidiš | melju | 7 | no parse |
| bujan | sedam | 6 | no parse |
| vučeš | voliš | 6 | no parse |
| pekao | uroni | 8 | no parse |

**Note:** Here **GPT** and **Gemini** tied on **relaxed** success rate (**5%**), but **neither** matched **optimal** length on those successes, and **graph-strict** validity stayed **0%**. **BERT** stayed **100% / 100%**.

---

## Croatian 4-letter — BERT vs GPT `gpt-5.4-mini` vs Gemini `gemini-2.5-flash`

**Setup:** `croatian_4`, **20 pairs** (`test4croatian`), notebook 13. **GPT** = `gpt-5.4-mini`, **Gemini** = `gemini-2.5-flash`.

### BERT (A\* only)

| Metric | Value |
|--------|-------|
| Wall time (20 pairs) | **346.6 s** (~**17.33 s** / pair) |
| Method | **A\* 20 / 20** |
| **Valid ladder** | **100%** |
| **Optimal length** (vs sampled BFS distance) | **100%** |

`df_bert.head()`:

| | start | target | bfs_optimal | method | steps | valid | optimal_length |
|---|-------|--------|-------------|--------|-------|-------|----------------|
| 0 | soda | duša | 3 | astar | 3 | True | True |
| 1 | besa | umre | 8 | astar | 8 | True | True |
| 2 | koga | kaki | 3 | astar | 3 | True | True |
| 3 | išta | gadi | 7 | astar | 7 | True | True |
| 4 | ulog | draž | 8 | astar | 8 | True | True |

### LLMs — summary row

| Metric | BERT A\* | GPT `gpt-5.4-mini` | Gemini `gemini-2.5-flash` |
|--------|----------|--------------------|---------------------------|
| Valid / success | **100%** | **10%** (2 / 20) | **15%** (3 / 20) |
| Optimal length (vs sampled opt) | **100%** | **0%** | **5%** |
| Mean steps (when success) | **5.30** | **3.50** | **3.33** |
| Total wall seconds | **346.6** | **14.1** | **55.5** |

### Gemini — diagnostics (same run)

| Metric | Value |
|--------|-------|
| Mean API latency / pair | **~2.77 s** |
| **Valid on island graph** (`valid_on_graph`) | **5%** (1 / 20) |
| Mean fraction of path words in graph | **~0.27** |
| Mean `optimal_length` among summary-**valid** rows | **~33%** (some relaxed-valid ladders were not shortest vs `bfs_optimal`) |

### GPT failures (sample)

| start | target | bfs_optimal | fail_reason (abridged) |
|-------|--------|-------------|-------------------------|
| soda | duša | 3 | not one-letter step: `suda` → `duša` |
| besa | umre | 8 | not one-letter step: `mesa` → `mrea` |
| koga | kaki | 3 | not one-letter step: `koga` → `kaka` |
| išta | gadi | 7 | not one-letter step: `gšta` → `gadi` |
| ulog | draž | 8 | not one-letter step: `ulog` → `ulog` |
| takt | umor | 8 | not one-letter step: `takt` → `trok` |
| šiva | prsi | 4 | not one-letter step: `siva` → `siva` |
| diše | ruda | 4 | not one-letter step: `ruse` → `ruda` |
| nađi | biro | 5 | not one-letter step: `bari` → `biro` |
| bina | dlan | 7 | not one-letter step: `dina` → `dlan` |
| izum | jaši | 10 | not one-letter step: `izum` → `jazm` |
| peka | saga | 4 | not one-letter step: `seka` → `seka` |
| bife | pušu | 5 | not one-letter step: `pife` → `puše` |
| srna | trik | 5 | not one-letter step: `trna` → `trik` |
| prsi | naum | 6 | not one-letter step: `prsi` → `prsi` |

### Gemini failures (sample)

| start | target | bfs_optimal | fail_reason (abridged) |
|-------|--------|-------------|-------------------------|
| bife | pušu | 5 | no parse |
| hrpa | tući | 4 | no parse |
| srna | trik | 5 | no parse |
| prsi | naum | 6 | no parse |

**Note:** On this slice **Gemini** beat **GPT** on relaxed validity (**15%** vs **10%**) and had **non-zero** **on-graph** success (**5%**), plus **5%** optimal vs BFS label — still far below **BERT** at **100% / 100%**.

---

## Changelog

- **2026-04-04:** English 5-letter, 20 pairs, `gpt-5.4-mini` vs BERT — metrics, BERT head sample, and GPT failure table logged above.
- **2026-04-04:** Same pairs, **`gpt-5.4`** (full) — 0% valid, failure sample, mini vs full comparison.
- **2026-04-04:** Same pairs, **`gpt-5.4`** + **`gemini-2.5-flash`** — three-way summary, Gemini diagnostics and failure sample.
- **2026-04-04:** English **4-letter**, 20 pairs, BERT vs **`gpt-5.4-mini`** vs **`gemini-2.5-flash`** — full summary and failure tables.
- **2026-04-04:** Croatian **5-letter**, 20 pairs — same LLMs; BERT 100% / 100%, GPT & Gemini 5% relaxed valid, 0% optimal / 0% on-graph.
- **2026-04-04:** Croatian **4-letter**, 20 pairs — BERT 100% / 100%; GPT 10% / 0% optimal; Gemini 15% relaxed, 5% on-graph, 5% optimal vs BFS label.
