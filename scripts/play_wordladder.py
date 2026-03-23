#!/usr/bin/env python3
"""
Play Word Ladder using the fine-tuned BERT distance-regression model.

At each step, scores every neighbor by predicted distance to the target
and picks the closest (A* heuristic). Falls back to BFS if beam search fails.

Usage:
  python scripts/play_wordladder.py crane flame
  python scripts/play_wordladder.py crane flame --beam 5
  python scripts/play_wordladder.py   # interactive mode
"""

import argparse
import string
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Project paths (run from repo root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "bert_wordladder_5letter"
VOCAB_PATH = PROJECT_ROOT / "data" / "islands" / "english_5_largest_island.txt"
MAX_LENGTH = 32


def load_vocab(path: Path) -> set[str]:
    """Load 5-letter words from file."""
    return {
        w.strip().lower()
        for w in path.read_text(encoding="utf-8").splitlines()
        if w.strip() and len(w.strip()) == 5
    }


def one_letter_neighbors(w: str, vocab: set[str]) -> set[str]:
    """Words that differ from w by exactly one letter."""
    return {
        w[:i] + c + w[i + 1 :]
        for i in range(len(w))
        for c in string.ascii_lowercase
        if c != w[i]
    } & vocab


def score_candidates(model, tokenizer, current: str, target: str, candidates: list, device):
    """Predict distance from each candidate to target (batched). Returns (candidate, dist) sorted ascending."""
    if not candidates:
        return []

    enc = tokenizer(
        [cand for cand in candidates],
        [target] * len(candidates),
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    model.eval()
    with torch.no_grad():
        out = model(**enc)
        preds = out.logits.squeeze(-1).cpu().tolist()

    if isinstance(preds, float):
        preds = [preds]

    scores = list(zip(candidates, preds))
    return sorted(scores, key=lambda x: x[1])


def shortest_path_bfs(start: str, target: str, vocab: set[str]) -> list[str] | None:
    """BFS fallback when greedy model gets stuck. Returns shortest path or None."""
    if start not in vocab or target not in vocab:
        return None
    from collections import deque
    queue = deque([(start, [start])])
    seen = {start}
    while queue:
        w, path = queue.popleft()
        if w == target:
            return path
        for n in one_letter_neighbors(w, vocab) - seen:
            seen.add(n)
            queue.append((n, path + [n]))
    return None


def generate_path_astar(
    model,
    tokenizer,
    vocab: set[str],
    start: str,
    target: str,
    max_expansions: int = 300,
    max_steps: int = 20,
    device=None,
) -> tuple[list[str] | None, str]:
    """
    A* search with BERT predicted distance as heuristic.
    Uses a priority queue so short paths with good heuristic values
    are expanded before longer detours.
    """
    import heapq

    if device is None:
        device = next(model.parameters()).device

    start = start.lower().strip()
    target = target.lower().strip()

    if start not in vocab or target not in vocab or len(start) != 5 or len(target) != 5:
        return None, None

    h0 = score_candidates(model, tokenizer, start, target, [start], device)[0][1]
    h0 = max(h0, 0)

    counter = 0
    pq = [(h0, counter, [start])]
    best_g = {start: 0}
    expansions = 0

    while pq and expansions < max_expansions:
        f, _, path = heapq.heappop(pq)
        current = path[-1]
        g_current = len(path) - 1

        if current == target:
            return path, "astar"

        if g_current >= max_steps:
            continue

        if g_current > best_g.get(current, float('inf')):
            continue

        neighbors = list(one_letter_neighbors(current, vocab))
        if not neighbors:
            continue

        ranked = score_candidates(model, tokenizer, current, target, neighbors, device)
        expansions += 1

        for word, pred_dist in ranked:
            g_new = g_current + 1
            if g_new < best_g.get(word, float('inf')):
                best_g[word] = g_new
                h = max(pred_dist, 0)
                f_score = g_new + h
                counter += 1
                heapq.heappush(pq, (f_score, counter, path + [word]))

    bfs_path = shortest_path_bfs(start, target, vocab)
    return (bfs_path, "bfs") if bfs_path else (None, None)


def generate_path(
    model,
    tokenizer,
    vocab: set[str],
    start: str,
    target: str,
    max_steps: int = 20,
    device=None,
    **kwargs,
) -> tuple[list[str] | None, str]:
    """Generate path using A* search with BERT heuristic. Falls back to BFS."""
    return generate_path_astar(
        model, tokenizer, vocab, start, target,
        max_steps=max_steps, device=device,
    )


def main():
    parser = argparse.ArgumentParser(description="Play Word Ladder with BERT model")
    parser.add_argument("start", nargs="?", help="Start word (5 letters)")
    parser.add_argument("target", nargs="?", help="Target word (5 letters)")
    parser.add_argument(
        "--model",
        default=str(MODEL_PATH),
        help=f"Path to model (default: {MODEL_PATH})",
    )
    parser.add_argument(
        "--vocab",
        default=str(VOCAB_PATH),
        help=f"Path to vocab (default: {VOCAB_PATH})",
    )
    args = parser.parse_args()

    # Interactive mode
    if not args.start or not args.target:
        print("Word Ladder — BERT-guided path between 5-letter words")
        print("(Run with: python scripts/play_wordladder.py START TARGET)")
        print()
        try:
            args.start = input("Start word: ").strip().lower()
            args.target = input("Target word: ").strip().lower()
        except EOFError:
            sys.exit(0)

    model_path = Path(args.model)
    vocab_path = Path(args.vocab)

    if not model_path.exists():
        print(f"Model not found at {model_path}")
        print("Download from Colab: zip models/bert_wordladder_5letter and extract here.")
        sys.exit(1)
    if not vocab_path.exists():
        print(f"Vocab not found at {vocab_path}")
        print("Run notebook 04 to generate island files.")
        sys.exit(1)

    print("Loading model and vocab...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = model.to(device)

    vocab = load_vocab(vocab_path)
    print(f"Vocab: {len(vocab)} words")
    print()

    path, method = generate_path(
        model, tokenizer, vocab, args.start, args.target,
        device=device,
    )

    if path:
        label = f" ({method})" if method else ""
        print(f"Path ({args.start} → {args.target}): {len(path)-1} steps{label}")
        print(" → ".join(path))
    else:
        print(f"No path found from '{args.start}' to '{args.target}'")
        print("(Check both words are in the 5-letter vocab and that a path exists)")


if __name__ == "__main__":
    main()
