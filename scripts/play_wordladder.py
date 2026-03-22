#!/usr/bin/env python3
"""
Play Word Ladder using the fine-tuned BERT model.

Generates a path from start → target by greedily picking the best-scoring
neighbor at each step.

Usage:
  python scripts/play_wordladder.py cat dog
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
MAX_LENGTH = 64


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
    """Score each candidate. Returns list of (candidate, score) sorted by score descending."""
    text_a = f"{current} [SEP] {target}"
    scores = []

    model.eval()
    with torch.no_grad():
        for cand in candidates:
            enc = tokenizer(
                text_a,
                cand,
                truncation=True,
                max_length=MAX_LENGTH,
                padding="max_length",
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            probs = torch.softmax(out.logits, dim=1)
            score = probs[0, 1].item()  # P(label=1)
            scores.append((cand, score))

    return sorted(scores, key=lambda x: -x[1])


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


def generate_path_beam(
    model,
    tokenizer,
    vocab: set[str],
    start: str,
    target: str,
    beam_size: int = 3,
    max_steps: int = 20,
    device=None,
) -> tuple[list[str] | None, str]:
    """
    Beam search: keep top-k paths at each step. Returns (path, "beam"|"bfs"|None).
    Falls back to BFS if all beams get stuck.
    """
    if device is None:
        device = next(model.parameters()).device

    start = start.lower().strip()
    target = target.lower().strip()

    if start not in vocab or target not in vocab or len(start) != 5 or len(target) != 5:
        return None, None

    beam = [([start], 1.0)]  # (path, last_step_score)

    for _ in range(max_steps - 1):
        candidates = []
        for path, _ in beam:
            if path[-1] == target:
                return path, "beam"
            visited = set(path)
            neighbors = one_letter_neighbors(path[-1], vocab) - visited
            if not neighbors:
                continue
            ranked = score_candidates(
                model, tokenizer, path[-1], target, list(neighbors), device
            )
            for word, score in ranked[:beam_size]:
                candidates.append((path + [word], score))

        if not candidates:
            break
        candidates.sort(key=lambda x: -x[1])
        beam = candidates[:beam_size]

    if beam and beam[0][0][-1] == target:
        return beam[0][0], "beam"
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
    fallback_bfs: bool = True,
    beam_size: int = 3,
) -> tuple[list[str] | None, str]:
    """
    Generate path using beam search (beam_size > 1) or greedy (beam_size=1).
    Returns (path, "beam"|"model"|"bfs"|None).
    """
    if beam_size > 1:
        return generate_path_beam(
            model, tokenizer, vocab, start, target,
            beam_size=beam_size, max_steps=max_steps, device=device
        )
    # Greedy (original logic)
    if device is None:
        device = next(model.parameters()).device
    start, target = start.lower().strip(), target.lower().strip()
    if start not in vocab or target not in vocab or len(start) != 5 or len(target) != 5:
        return None, None
    path, visited = [start], {start}
    for _ in range(max_steps - 1):
        if path[-1] == target:
            return path, "model"
        neighbors = one_letter_neighbors(path[-1], vocab) - visited
        if not neighbors:
            break
        ranked = score_candidates(model, tokenizer, path[-1], target, list(neighbors), device)
        path.append(ranked[0][0])
        visited.add(path[-1])
    if path[-1] == target:
        return path, "model"
    if fallback_bfs:
        bfs_path = shortest_path_bfs(start, target, vocab)
        return (bfs_path, "bfs") if bfs_path else (None, None)
    return None, None


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
    parser.add_argument(
        "--beam", type=int, default=3,
        help="Beam size (default: 3). Use 1 for greedy.",
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
        device=device, beam_size=args.beam,
    )

    if path:
        label = f" ({method})" if method else ""
        print(f"Path ({args.start} → {args.target}): {len(path)} steps{label}")
        print(" → ".join(path))
    else:
        print(f"No path found from '{args.start}' to '{args.target}'")
        print("(Check both words are in the 5-letter vocab and that a path exists)")


if __name__ == "__main__":
    main()
