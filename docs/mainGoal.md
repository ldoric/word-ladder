# Word Ladder Solver – BERT vs LLMs  
**Diplomski rad / Master's thesis project – 2026**

## Goal (one sentence)
Build and compare neural solvers for the **Word Ladder** game (change one letter at a time, every step must be a valid word) using a fine-tuned **BERT** model vs frontier **LLMs** (GPT-4o, Claude 3.5/Opus, Gemini 1.5, etc.), showing that BERT-guided search is more reliable and rule-adherent.

## Core Idea
- Word Ladder: e.g. COLD → CORD → CARD → WARD → WARM (or Croatian: hladno → toplo variants)
- Build graph of valid 4-letter and 5-letter words (English + Croatian)
- Fine-tune small BERT (bert-base-multilingual-cased or croatian-bert) to score "best next word" given current word + target
- Use beam search / A* with BERT as heuristic → find shortest valid paths
- Compare vs zero/few-shot LLMs → LLMs often fail by:
  - changing >1 letter
  - inventing non-words
  - looping / repeating
  - high failure rate on longer/harder ladders
- Croatian version = major novelty (first serious Slavic/South-Slavic Word Ladder benchmark with neural guidance)

## Deliverables
- Clean word lists → english_4.txt, english_5.txt, croatian_4.txt, croatian_5.txt
- Graph (networkx) + BFS-generated training data for BERT
- Fine-tuned BERT scorer + guided search solver
- LLM baseline evaluation (API calls with strict prompting)
- Metrics on 100s–1000s test pairs: success rate, avg path length, rule adherence, speed
- Simple web demo (Gradio / Streamlit): input start/target → see path + stats
- BERT “hint” API: **`word-ladder-api/`** (FastAPI, Hugging Face Space) — per-step neighbor suggestion for the game; public URL pattern `https://<user>-<space>.hf.space` (see `docs/context.md`)
- Thesis sections: related work, methodology, results, Croatian originality point

## Why BERT wins (thesis one-liner)
Fine-tuned BERT excels at precise single-letter neighborhood prediction due to bidirectional context and task-specific training, while autoregressive LLMs struggle with strict rule-following and systematic exploration.

## Folder structure (so far)
- data/raw/          ← original downloaded lists
- data/processed/    ← cleaned & filtered .txt files (4/5 letters)
- notebooks/         ← Jupyter notebooks (e.g. 01_word_lists_combine.ipynb)
- word-ladder-api/  ← FastAPI BERT hint service (Docker) for the web app; can be split to its own git repo
- src/               ← future .py modules (graph.py, solver.py, bert_finetune.py)
- requirements.txt

Status: Virtual env + VS Code Jupyter notebooks working. Next: combine & clean English word lists, then Croatian.

Professor-approved task summary (Croatian):  
"Izrada modela za rješavanje igre Word Ladder koristeći finetunirani BERT model... graf susjednih riječi na engleskom i hrvatskom... usporediti s LLM-ovima... web sučelje... testirati na velikom broju parova."

Let's build something reproducible and publishable!