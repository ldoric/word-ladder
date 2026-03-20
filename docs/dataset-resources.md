# Dataset Resources and Build Formula

This document lists all sources used to build the current Word Ladder datasets.

## Raw resources used
- Donald Knuth SGB words (5-letter base)
  - https://cs.stanford.edu/~knuth/sgb-words.txt
  - local: data/raw/sgb-words.txt
- dwyl English words
  - https://raw.githubusercontent.com/dwyl/english-words/master/words.txt
  - local: data/raw/dwyl_words.txt
- Burkardt doublet words
  - https://people.sc.fsu.edu/~jburkardt/datasets/words/doublet_words.txt
  - local: data/raw/doublet_words.txt
- Burkardt knuth words mirror/variant
  - https://people.sc.fsu.edu/~jburkardt/datasets/words/knuth_words.txt
  - local: data/raw/knuth_words.txt
- Wordle valid guesses (dracos gist)
  - https://gist.githubusercontent.com/dracos/dd0668f281e685bad51479e5acaadb93/raw/valid-wordle-words.txt
  - local: data/raw/valid-wordle-words.txt
- 4-letter list (paulcc gist)
  - https://gist.githubusercontent.com/paulcc/3799331/raw/
  - local: data/raw/paulcc_4letter_words.txt
- 4-letter processed list (raspberrypisig gist)
  - https://gist.githubusercontent.com/raspberrypisig/cc18b0f4fbc0c79ffd667d06adc0a190/raw/4-letter-words-processed-new.txt
  - local: data/raw/rpi_4letter_words.txt
- 4-letter Scrabble list (rmmh/fourletterword)
  - https://raw.githubusercontent.com/rmmh/fourletterword/master/list.txt
  - local: data/raw/rmmh_4letter_words.txt

## Normalization
For every source line:
- trim whitespace
- lowercase
- keep only ASCII alphabetic tokens
- filter by exact length (4 or 5)
- deduplicate via set operations

## Build formulas (current)
Let:
- SGB5 = 5-letter set from sgb-words.txt
- WV5 = 5-letter set from valid-wordle-words.txt
- D5, DBL5, K5, W35 = 5-letter sets from dwyl, doublet, knuth_words, wordl3
- D4, DBL4, K4, W34 = 4-letter sets from dwyl, doublet, knuth_words, wordl3
- P4 = 4-letter set from paulcc
- R4 = 4-letter set from raspberrypisig
- M4 = 4-letter set from rmmh

Then:
- english_5_strict = SGB5 union { w | w in at least 2 of [D5, DBL5, K5, W35] }
- english_5 = SGB5 union { w | w in WV5 and w in at least 1 of [D5, DBL5, K5, W35] }
- english_4_strict = { w | w in at least 2 of [DBL4, P4, R4, M4] }
- english_4 = union(DBL4, P4, R4, M4)

## Outputs
- data/english_5_strict.txt
- data/english_5.txt
- data/english_4_strict.txt
- data/english_4.txt
- data/croatian_5_strict.txt
- data/croatian_5.txt
- data/croatian_4_strict.txt
- data/croatian_4.txt
- data/review_croatian_4.txt (generated for manual review)
- data/review_croatian_5.txt (generated for manual review)
- data/islands/english_4_largest_island.txt
- data/islands/english_4_strict_largest_island.txt
- data/islands/english_4_strict_only_island.txt
- data/islands/english_5_largest_island.txt
- data/islands/english_5_strict_largest_island.txt
- data/islands/english_5_strict_only_island.txt
- data/islands/croatian_4_largest_island.txt
- data/islands/croatian_4_strict_largest_island.txt
- data/islands/croatian_4_strict_only_island.txt
- data/islands/croatian_5_largest_island.txt
- data/islands/croatian_5_strict_largest_island.txt
- data/islands/croatian_5_strict_only_island.txt

## Croatian resources (2026)
- Rijecalica (game-cleaned wordlist, best quality)
  - https://raw.githubusercontent.com/Martinsos/Rijecalica/master/croatian-wordlist-checked-iso8859-2.txt
  - local: data/raw/rijecalica_croatian.txt (converted to UTF-8 on download)
  - ISO-8859-2 source; expected 1–3k 4/5-letter words
- kkrypt0nn/wordlists Croatian (large & comprehensive)
  - https://raw.githubusercontent.com/kkrypt0nn/wordlists/main/wordlists/languages/croatian.txt
  - local: data/raw/kkrypt0nn_croatian.txt
  - UTF-8, ~95k entries, includes inflected forms
- gigaly/rjecnik-hrvatskih-jezika HR_Txt (624k+ words)
  - https://github.com/gigaly/rjecnik-hrvatskih-jezika/releases → HR_Txt-624.zip
  - local: data/raw/HR_Txt-624.txt (user-provided)
  - Tab-separated: word\tlemma\t... ; we use first column
- ePADD/muse en-abbreviations (English abbreviations/acronyms)
  - https://raw.githubusercontent.com/ePADD/muse/master/WebContent/WEB-INF/classes/dictionaries/en-abbreviations.txt
  - local: data/raw/en-abbreviations.txt (downloaded on first run)
  - Used to remove acronyms from strict Croatian lists (NASA, FBI, etc.)

## Croatian normalization
- lowercase
- keep Croatian letters (č, ć, đ, š, ž)
- filter by exact length (4 or 5)
- isalpha() for valid words

## Croatian build formulas
Let R4, K4, H4 = 4-letter sets from Rijecalica, kkrypt0nn, HR_Txt.
Let R5, K5, H5 = 5-letter sets from same sources.
- croatian_4_strict = R4 ∩ K4 ∩ H4 (all 3 sources) minus blocklist, **nominative + glagol only**
- croatian_5_strict = R5 ∩ K5 ∩ H5 (all 3 sources) minus blocklist, **nominative + glagol only**
- croatian_4 = { w | w in at least 2 of [R4, K4, H4] } minus blocklist
- croatian_5 = { w | w in at least 2 of [R5, K5, H5] } minus blocklist

**Strict filter (nominative + glagol):** Uses HR_Txt lemma. Keeps only words where `word == lemma` (nominative/base form) or POS is `glagol` (verb forms, incl. past participle). Removes inflected noun/adjective forms (genitive, dative, etc.).

Acronym blocklist: filters known abbreviations (adcp, adsl, etc.); extend in notebook. Strict lists also remove matches from ePADD/muse en-abbreviations.txt (see notebook cell).

**Manual acronym removals (paper-reproducible):** Seven words were manually excluded from strict lists because they match English abbreviations/acronyms (e.g. CRES/Crescent, STAT/station, TRAK/track, NERV/nerve, TROP/tropical, MILIT/military, POLIT/political) and may cause ambiguity in Word Ladder. Removed from croatian_4_strict: *cres*, *stat*, *trak*, *nerv*, *trop*. Removed from croatian_5_strict: *milit*, *polit*.

## Croatian manual review workflow

Both strict Croatian files have been manually removed and filtered.

**Strict 4:** Manually curated – all unwanted words have been removed from croatian_4_strict.

**Strict 5:** Manually curated – all unwanted words have been removed from croatian_5_strict.

1. Run "Generate review files" cell → heuristic-flagged candidates; or run helper cells (e.g. h-ending) → targeted subsets
2. Edit review files: DELETE lines for words to keep, LEAVE only words to remove
3. Run "Apply manual removals" cell → removes those words from **strict only** (croatian_4_strict, croatian_5_strict). Non-strict files (croatian_4, croatian_5) are unchanged.

## English island files (largest component only)

Derived from notebooks/04_english_datasets_testing.ipynb. For gameplay, only words in the largest connected component are used so every start/end pair is solvable.

| File | Steps | Endpoints | 4-letter | 5-letter |
|------|-------|-----------|----------|----------|
| english_n_largest_island.txt | non-strict | non-strict | 5643 | 9902 |
| english_n_strict_largest_island.txt | non-strict | strict | 3155 | 5330 |
| english_n_strict_only_island.txt | strict | strict | 3050 | 4573 |

## Croatian island files (largest component only)

Derived from notebooks/03_croatian_datasets_testing.ipynb. For gameplay, only words in the largest connected component are used so every start/end pair is solvable.

| File | Steps | Endpoints | 4-letter | 5-letter |
|------|-------|-----------|----------|----------|
| croatian_n_largest_island.txt | non-strict | non-strict | 3863 | 11052 |
| croatian_n_strict_largest_island.txt | non-strict | strict | 941 | 1448 |
| croatian_n_strict_only_island.txt | strict | strict | 767 | 57 |

## Implementation location
- notebooks/01_word_lists_combine.ipynb – English (Cell 4)
- notebooks/02_croatian_wordlists.ipynb – Croatian (Cell 4)
- notebooks/03_croatian_datasets_testing.ipynb – Croatian islands (filter to largest component)
- notebooks/04_english_datasets_testing.ipynb – English islands (filter to largest component)
