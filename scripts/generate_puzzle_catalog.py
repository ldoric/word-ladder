#!/usr/bin/env python3
"""
Generate data/puzzles/catalog.json from strict-largest island word lists.

Graph: vertices = words in the island file; edges = one-letter substitutions
staying in the vocabulary. Solutions are one BFS shortest path (lexicographic
tie-break on the first branch).

Difficulty buckets (BFS edge count = len(path) - 1):
  easy:   3–5 edges (four puzzles: distances 3,4,4,5)
  medium: 6–7 edges (four puzzles: 6,6,7,7)
  hard:   8 edges (four puzzles)
"""

from __future__ import annotations

import argparse
import json
import random
from collections import deque
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ISLANDS = REPO_ROOT / "data" / "islands"
DEFAULT_OUT = REPO_ROOT / "data" / "puzzles" / "catalog.json"

CONFIGS = [
    ("english", 4, ISLANDS / "english_4_strict_largest_island.txt"),
    ("english", 5, ISLANDS / "english_5_strict_largest_island.txt"),
    ("croatian", 4, ISLANDS / "croatian_4_strict_largest_island.txt"),
    ("croatian", 5, ISLANDS / "croatian_5_strict_largest_island.txt"),
]

DISTANCE_PLAN = {
    "easy": [3, 4, 4, 5],
    "medium": [6, 6, 7, 7],
    "hard": [8, 8, 8, 8],
}


def load_word_set(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8")
    return {ln.strip().lower() for ln in text.splitlines() if ln.strip()}


def charset_for(words: set[str]) -> list[str]:
    chars = set()
    for w in words:
        chars.update(w)
    return sorted(chars)


def neighbors(word: str, words: set[str], chars: list[str]) -> list[str]:
    L = len(word)
    out: list[str] = []
    for i in range(L):
        for c in chars:
            if c == word[i]:
                continue
            nw = word[:i] + c + word[i + 1:]
            if nw in words:
                out.append(nw)
    return out


def bfs_path(start: str, target: str, words: set[str], chars: list[str]) -> list[str] | None:
    if start == target:
        return [start]
    if start not in words or target not in words:
        return None
    parent: dict[str, str | None] = {start: None}
    q: deque[str] = deque([start])
    while q:
        w = q.popleft()
        for nb in sorted(neighbors(w, words, chars)):
            if nb not in parent:
                parent[nb] = w
                if nb == target:
                    path = [target]
                    cur = target
                    while cur != start:
                        cur = parent[cur]  # type: ignore[assignment]
                        path.append(cur)
                    path.reverse()
                    return path
                q.append(nb)
    return None


def nodes_at_distance(start: str, words: set[str], chars: list[str], d: int) -> list[str]:
    """All nodes at shortest-path distance exactly d from start."""
    if d < 0:
        return []
    if d == 0:
        return [start]
    dist: dict[str, int] = {start: 0}
    q: deque[str] = deque([start])
    while q:
        w = q.popleft()
        dw = dist[w]
        if dw == d:
            continue
        for nb in neighbors(w, words, chars):
            if nb not in dist:
                dist[nb] = dw + 1
                if dist[nb] <= d:
                    q.append(nb)
    return sorted(w for w, k in dist.items() if k == d)


def pick_pair_at_distance(
    words: set[str],
    chars: list[str],
    d: int,
    used: set[tuple[str, str]],
    rng: random.Random,
) -> tuple[str, str, list[str]] | None:
    """Return (start, target, path) or None."""
    candidates_starts = list(words)
    rng.shuffle(candidates_starts)
    for start in candidates_starts:
        layer = nodes_at_distance(start, words, chars, d)
        if not layer:
            continue
        rng.shuffle(layer)
        for target in layer:
            a, b = (start, target), (target, start)
            if a in used or b in used:
                continue
            path = bfs_path(start, target, words, chars)
            if path is None:
                continue
            if len(path) - 1 != d:
                continue
            return start, target, path
    return None


def display_word(w: str) -> str:
    return w.upper()


def subtitle(start: str, target: str) -> str:
    return f"{display_word(start)} → {display_word(target)}"


def puzzle_id(start: str, target: str, used_ids: set[str]) -> str:
    """Stable slug: firstword-lastword (lowercase, as in island files)."""
    base = f"{start}-{target}"
    if base not in used_ids:
        used_ids.add(base)
        return base
    n = 2
    while True:
        cand = f"{base}-{n}"
        if cand not in used_ids:
            used_ids.add(cand)
            return cand
        n += 1


def build_catalog(seed: int) -> dict:
    rng = random.Random(seed)
    puzzles: list[dict] = []
    used_ids: set[str] = set()
    for locale, n_letters, island_path in CONFIGS:
        if not island_path.is_file():
            raise FileNotFoundError(island_path)
        words = load_word_set(island_path)
        chars = charset_for(words)
        used: set[tuple[str, str]] = set()
        for difficulty in ("easy", "medium", "hard"):
            for d in DISTANCE_PLAN[difficulty]:
                found = pick_pair_at_distance(words, chars, d, used, rng)
                if found is None:
                    raise RuntimeError(
                        f"No pair at distance {d} for {island_path.name} ({locale} {n_letters}). "
                        "Try a different seed or relax distances."
                    )
                start, target, path = found
                used.add((start, target))
                pid = puzzle_id(start, target, used_ids)
                puzzles.append(
                    {
                        "id": pid,
                        "locale": locale,
                        "wordLength": n_letters,
                        "difficulty": difficulty,
                        "bfsEdges": d,
                        "subtitle": subtitle(start, target),
                        "solution": [display_word(w) for w in path],
                    }
                )
    return {"schemaVersion": 1, "puzzles": puzzles}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output JSON path")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for pair selection")
    args = p.parse_args()
    catalog = build_catalog(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(catalog['puzzles'])} puzzles to {args.out}")


if __name__ == "__main__":
    main()
