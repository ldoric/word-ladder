#!/usr/bin/env python3
"""
Filter Croatian word lists to keep only nominative/base forms.

Uses HR_Txt dictionary: when word == lemma, the word is in its dictionary form
(nominative for nouns, base form for adjectives, etc.). Inflected forms
(genitive, dative, accusative, etc.) have word != lemma.
"""
from pathlib import Path


def load_nominative_forms(hr_txt_path: Path, word_len: int) -> set[str]:
    """Load words of length word_len where word == lemma (nominative/base form)."""
    if not hr_txt_path.exists():
        return set()
    nominative = set()
    for line in hr_txt_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split("\t")
        if len(parts) < 2:
            continue
        word = parts[0].strip().lower()
        lemma = parts[1].strip().lower()
        if len(word) == word_len and word.isalpha() and word == lemma:
            nominative.add(word)
    return nominative


def filter_to_nominative(
    word_list_path: Path,
    hr_txt_path: Path,
    word_len: int = 4,
    output_path: Path | None = None,
) -> tuple[list[str], int, int]:
    """
    Filter word list to keep only nominative forms.
    Returns (filtered_words, original_count, removed_count).
    """
    nominative = load_nominative_forms(hr_txt_path, word_len)
    words = [
        w.strip()
        for w in word_list_path.read_text(encoding="utf-8").splitlines()
        if w.strip()
    ]
    original = len(words)
    filtered = [w for w in words if w in nominative]
    removed = original - len(filtered)
    out = output_path or word_list_path
    out.write_text("\n".join(filtered) + ("\n" if filtered else ""), encoding="utf-8")
    return filtered, original, removed


def main():
    base = Path(__file__).resolve().parent.parent
    hr_txt = base / "data" / "raw" / "HR_Txt-624.txt"

    for name, word_len in [("croatian_4_strict.txt", 4), ("croatian_5_strict.txt", 5)]:
        path = base / "data" / name
        if not path.exists():
            print(f"Skip (missing): {path.name}")
            continue
        words, orig, removed = filter_to_nominative(path, hr_txt, word_len=word_len)
        print(f"{name}: {orig} -> {len(words)} words (removed {removed} inflected forms)")
    print("Kept only nominative/base forms (word == lemma in HR_Txt)")


if __name__ == "__main__":
    main()
