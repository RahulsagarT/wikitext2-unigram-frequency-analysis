# Unigram Frequency Analysis of Wikipedia Articles (WikiText-2)

Builds a unigram (single-word) language model from the WikiText-2 training set.

## What it does
1. Reads the WikiText-2 **training** split
2. Lowercases the text and tokenizes it (letters, digits, internal apostrophes; punctuation and `@-@` markers are dropped)
3. Counts the frequency of every word
4. Lists the 20 most frequent words
5. Computes unigram probabilities for five selected words:
   `P(w) = count(w) / total number of tokens`
6. Saves a word-frequency table (CSV) and a bar chart of the top 20 words

## Project structure
```
.
├── unigram_wikitext.py
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
└── output/                  # created on first run
    ├── word_frequencies.csv
    └── top20_words.png
```

## Installation
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage
```bash
python unigram_wikitext.py                                  # downloads WikiText-2 via Hugging Face
python unigram_wikitext.py --words the of and university war   # choose your own five words
python unigram_wikitext.py --file wiki.train.tokens         # use a local copy (offline)
```

## Results
Run the script and paste your output here.

**Top 20 words**

| Rank | Word | Frequency | Probability |
|------|------|-----------|-------------|
|      |      |           |             |

**Selected words**

| Word | Count | P(w) |
|------|-------|------|
|      |       |      |

**Bar chart:** `![Top 20 words](output/top20_words.png)`

## Notes
- Words never seen in training get probability 0 (no smoothing is applied).
- Probabilities sum to 1 over the whole vocabulary.
- WikiText-2 is licensed CC BY-SA; the data is not included in this repo.

## License
MIT
