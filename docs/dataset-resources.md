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

## Implementation location
- notebooks/01_word_lists_combine.ipynb
- generation logic in Cell 5
