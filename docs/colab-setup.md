# Google Colab Setup

How to run the Word Ladder notebooks (05, 06) from Google Drive in Colab.

---

## 1. Clone the project into Google Drive

1. Open [Google Colab](https://colab.research.google.com)
2. Create a new notebook
3. **Runtime → Change runtime type → GPU (T4)** → Save

4. In the first cell, run:

```python
from google.colab import drive
drive.mount('/content/drive')
```

5. When prompted, authorize access to your Google Drive.

6. Clone the repo into your Drive (change `YOUR_USERNAME` to your GitHub username):

```python
!git clone https://github.com/ldoric/word-ladder.git /content/drive/MyDrive/ColabProjects/word-ladder
```

Create the parent folder first if needed:

```python
!mkdir -p /content/drive/MyDrive/ColabProjects
!git clone https://github.com/YOUR_USERNAME/word-ladder.git /content/drive/MyDrive/ColabProjects/word-ladder
```

---

## 2. Open notebooks from Google Drive

1. In Colab: **File → Open notebook**
2. Open the **Google Drive** tab
3. Go to `My Drive → ColabProjects → word-ladder → notebooks`
4. Open `05_english_5_letter_training.ipynb` or `06_bert_wordladder_finetune.ipynb`

---

## 3. Add setup cells (before running)

Before running any notebook cells, add these **two cells at the very top** of the notebook:

**Cell 1 – Mount Drive**

```python
from google.colab import drive
drive.mount('/content/drive')
```

**Cell 2 – Set project root**

```python
import os
from google.colab import drive

# Mount Drive (run only if not already mounted)
drive.mount('/content/drive', force_remount=False)  # force_remount=True only if needed

# CHANGE THIS to match your exact folder structure
# Example: /content/drive/MyDrive/ColabProjects/word-ladder
PROJECT_ROOT = "/content/drive/MyDrive/ColabProjects/word-ladder/notebooks"

# Switch to the project folder so relative paths work
os.chdir(PROJECT_ROOT)

# Optional: print to confirm
print("Current working directory:", os.getcwd())
print("Files in current directory:", os.listdir('.'))
```

Adjust `PROJECT_ROOT` if your folder is in a different location (e.g. `MyDrive/word-ladder/notebooks`).

---

## 4. Install dependencies

Add and run a cell:

```python
!pip install transformers datasets networkx accelerate torch
```

---

## 5. Run the notebooks

1. Run **notebook 05** first (generates training CSVs)
2. Run **notebook 06** second (fine-tunes BERT for distance regression)

Use **Runtime → Run all** or run cells manually (Shift+Enter).

### Run 7+ defaults (larger data)

- **05:** ~**600k** rows, **4000** BFS sources, weighted sampling (more short graph distances). Data gen may take **tens of minutes** on Colab (more BFS passes).
- **06:** **6 epochs**, cosine LR, 6% warmup. Training may take **several hours** on a T4 with 600k examples; if OOM, set `BATCH_SIZE = 16` in notebook 06.

After changing 05 or 06, **always regenerate CSVs (05) before retraining (06)** so the model matches the new data.

---

## 6. Save outputs (optional)

Model and CSVs are written to `models/` and `data/training/`. They stay in your Drive if the project lives under `/content/drive/MyDrive/`. To copy elsewhere:

```python
!cp -r models/ /content/drive/MyDrive/
!cp -r data/training/ /content/drive/MyDrive/
```

---

## 7. Download model for local use

To run the play script locally, download the trained model from Colab:

1. **Where is the model?** After training it should be in `models/bert_wordladder_5letter/`. If that folder is missing `model.safetensors`, it may be in `outputs/bert_wordladder/checkpoint-4221/` (or the highest checkpoint number).
2. **Verify in Colab** — run:
   ```python
   !ls -la ../models/bert_wordladder_5letter/
   !ls -la ../outputs/bert_wordladder/
   ```
   Download the folder that has `model.safetensors` (~500 MB).
3. **Zip and download:**
   ```python
   !zip -r model.zip models/bert_wordladder_5letter  # or outputs/bert_wordladder/checkpoint-4221
   ```
   Then Files → model.zip → Download.
4. Extract into your local repo so `models/bert_wordladder_5letter/model.safetensors` exists.
5. Run: `python scripts/play_wordladder.py saned scrip`

### If model.safetensors is missing in models/

Run this in Colab to copy from the latest checkpoint, then zip:

```python
import shutil
from pathlib import Path
OUTPUTS = Path("../outputs/bert_wordladder")
SAVE = Path("../models/bert_wordladder_5letter")
SAVE.mkdir(parents=True, exist_ok=True)
ckpts = sorted(OUTPUTS.glob("checkpoint-*"))
if ckpts:
    for f in ckpts[-1].iterdir():
        shutil.copy(f, SAVE / f.name)
    print("Copied from", ckpts[-1], "- zip models/bert_wordladder_5letter and download")
```

---

## OpenAI API key (notebook 13 — LLM vs BERT)

**Local (recommended):**

1. Copy the example file in the repo root:
   - `cp .env.local.example .env.local` (macOS/Linux)
   - Or copy `.env.local.example` to `.env.local` in File Explorer (Windows).
2. Edit `.env.local` and set `OPENAI_API_KEY=sk-...`. For **Gemini** in notebook 13, add `GEMINI_API_KEY=...` (Google AI Studio).
3. `.env.local` is **gitignored**; only `.env.local.example` (no real secret) is tracked.

Notebook 13 loads `.env.local` via `python-dotenv` before reading API keys from the environment. You can also set variables in the shell instead:

- PowerShell: `$env:OPENAI_API_KEY="sk-..."` ; `$env:GEMINI_API_KEY="..."`
- bash: `export OPENAI_API_KEY=sk-...` ; `export GEMINI_API_KEY=...`

**Colab:** use a Colab secret or paste into the config cell (`OPENAI_API_KEY` / `GEMINI_API_KEY`) for that session only — do not save the notebook with secrets in it.
