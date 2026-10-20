"""
Unigram Frequency Analysis of Wikipedia Articles (WikiText-2)
-------------------------------------------------------------
1. Reads the WikiText-2 training split
2. Lowercases and tokenizes the text
3. Counts the frequency of every word
4. Reports the 20 most frequent words
5. Computes unigram probabilities  P(w) = count(w) / total_tokens
6. Saves a word-frequency table (CSV) and a bar chart of the top 20 words

Usage:
    pip install -r requirements.txt
    python unigram_wikitext.py
    python unigram_wikitext.py --words the of and university war      # choose your own 5 words
    python unigram_wikitext.py --file wiki.train.tokens               # offline / local file
"""
import argparse
import csv
import os
import re
from collections import Counter

import matplotlib
matplotlib.use("Agg")           # works without a display
import matplotlib.pyplot as plt

DEFAULT_WORDS = ["the", "of", "and", "university", "war"]

# words made of letters (optionally with an internal apostrophe, e.g. "don't") or digits
TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)*|\d+")


# ----------------------------------------------------------------------
# 1. Read training data
# ----------------------------------------------------------------------
def load_wikitext2_train(local_file=None):
    if local_file:
        with open(local_file, encoding="utf-8") as f:
            return f.read()

    from datasets import load_dataset
    last_err = None
    for name in ("Salesforce/wikitext", "wikitext"):
        try:
            ds = load_dataset(name, "wikitext-2-raw-v1", split="train")
            return "\n".join(ds["text"])
        except Exception as e:           # try next mirror
            last_err = e
    raise RuntimeError(
        f"Could not load WikiText-2 ({last_err}). "
        "Download wiki.train.tokens and pass it with --file."
    )


# ----------------------------------------------------------------------
# 2. Lowercase + tokenize
# ----------------------------------------------------------------------
def tokenize(text):
    text = text.lower()
    text = text.replace("<unk>", " ")    # placeholder used in the non-raw version
    return TOKEN_RE.findall(text)        # drops punctuation, "@-@", "= Heading =" markers


# ----------------------------------------------------------------------
# 3-5. Frequencies and probabilities
# ----------------------------------------------------------------------
def unigram_probability(word, counts, total):
    return counts.get(word.lower(), 0) / total


def save_frequency_table(counts, total, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "word", "frequency", "probability"])
        for rank, (word, c) in enumerate(counts.most_common(), 1):
            w.writerow([rank, word, c, f"{c / total:.8f}"])


def plot_top_n(top, path, n=20):
    words = [w for w, _ in top][::-1]          # reversed so rank 1 is at the top
    freqs = [c for _, c in top][::-1]
    plt.figure(figsize=(9, 7))
    bars = plt.barh(words, freqs, color="steelblue")
    for bar, f in zip(bars, freqs):
        plt.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                 f" {f:,}", va="center", fontsize=8)
    plt.xlabel("Frequency")
    plt.ylabel("Word")
    plt.title(f"Top {n} Most Frequent Words - WikiText-2 (train)")
    plt.xlim(0, max(freqs) * 1.12)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", help="local WikiText-2 training file (optional)")
    ap.add_argument("--words", nargs=5, default=DEFAULT_WORDS,
                    help="five words whose unigram probability is reported")
    ap.add_argument("--outdir", default="output")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    text = load_wikitext2_train(args.file)
    tokens = tokenize(text)
    counts = Counter(tokens)
    total = len(tokens)

    print(f"Total tokens        : {total:,}")
    print(f"Vocabulary (unique) : {len(counts):,}")

    # Top 20 word-frequency table
    top20 = counts.most_common(20)
    print("\nTop 20 most frequent words")
    print(f"{'Rank':<6}{'Word':<14}{'Frequency':>12}{'Probability':>14}")
    print("-" * 46)
    for rank, (w, c) in enumerate(top20, 1):
        print(f"{rank:<6}{w:<14}{c:>12,}{c / total:>14.6f}")

    # Unigram probabilities for five selected words
    print("\nUnigram probabilities of selected words:  P(w) = count(w) / total_tokens")
    print(f"{'Word':<14}{'Count':>12}{'Probability':>16}")
    print("-" * 42)
    for w in args.words:
        c = counts.get(w.lower(), 0)
        print(f"{w:<14}{c:>12,}{unigram_probability(w, counts, total):>16.8f}")

    # Save outputs
    csv_path = os.path.join(args.outdir, "word_frequencies.csv")
    png_path = os.path.join(args.outdir, "top20_words.png")
    save_frequency_table(counts, total, csv_path)
    plot_top_n(top20, png_path)
    print(f"\nSaved full frequency table -> {csv_path}")
    print(f"Saved bar chart            -> {png_path}")


if __name__ == "__main__":
    main()
