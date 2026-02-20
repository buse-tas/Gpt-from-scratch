"""
Task 1 — BPE Tokenizer
Word-internal Byte Pair Encoding with </w> sentinel.
Trained on TRAIN split only to prevent dev/test leakage.

Design notes (from LLM theory):
- BPE learns a fixed vocabulary of subword tokens via greedy merge rules.
- </w> sentinel ensures merges never cross word boundaries.
- k merges is the only hyperparameter; larger k → fewer tokens per word
  (word_as_token_rate ↑) but diminishing returns above ~1600.
"""

import os
import re
from collections import Counter
from typing import List, Tuple, Iterable, Dict

WORD_END = "</w>"
_wsre    = re.compile(r"\s+")


# ── Normalization ────────────────────────────────────────────────────────────
def words_from_text_norm(text: str, normalization: str = "standard") -> List[str]:
    """
    Convert raw text to word list using one of two normalization schemes.
    - 'standard':         lowercase + extract [a-z]+ sequences only
    - 'aggressive_clean': lowercase + keep apostrophes, collapse whitespace
    Train on TRAIN only — apply the same normalization to every split downstream.
    """
    text = text.lower()
    if normalization == "standard":
        return re.findall(r"[a-z]+", text)
    elif normalization == "aggressive_clean":
        text = re.sub(r"https?://\S+|[\w.+-]+@[\w-]+\.\w+", " ", text)
        text = re.sub(r"[^a-z' ]+", " ", text)
        return [w.strip("'") for w in _wsre.split(text) if w.strip("'")]
    else:
        raise ValueError(f"Unknown normalization: {normalization}")


def words_from_file_norm(path: str, normalization: str = "standard") -> List[str]:
    with open(path, encoding="utf-8") as f:
        return words_from_text_norm(f.read(), normalization)


# ── BPE primitives ───────────────────────────────────────────────────────────
def word_to_symbols(word: str) -> Tuple[str, ...]:
    """'hello' → ('h', 'e', 'l', 'l', 'o', '</w>')"""
    return tuple(list(word) + [WORD_END])


def corpus_to_symbol_sequences(words: Iterable[str]) -> List[Tuple[str, ...]]:
    return [word_to_symbols(w) for w in words]


def get_pair_counts(seqs: List[Tuple[str, ...]]) -> Counter:
    """Count every adjacent (a, b) pair across all symbol sequences."""
    counts: Counter = Counter()
    for seq in seqs:
        for a, b in zip(seq, seq[1:]):
            counts[(a, b)] += 1
    return counts


def apply_merge(seqs: List[Tuple[str, ...]], a: str, b: str) -> List[Tuple[str, ...]]:
    """Replace every occurrence of adjacent (a, b) with the merged token a+b."""
    merged = a + b
    new_seqs = []
    for seq in seqs:
        out, i = [], 0
        while i < len(seq):
            if i < len(seq) - 1 and seq[i] == a and seq[i + 1] == b:
                out.append(merged)
                i += 2
            else:
                out.append(seq[i])
                i += 1
        new_seqs.append(tuple(out))
    return new_seqs


# ── BPETokenizer ─────────────────────────────────────────────────────────────
class BPETokenizer:
    """
    Word-internal BPE with </w> sentinel.
    Fit on TRAIN split only — no dev/test leakage.

    Key insight: merge rules are learned greedily by frequency.
    The vocabulary that emerges after k merges directly determines
    the average tokens-per-word and the word-as-token rate.
    """

    def __init__(self):
        self.merges:        List[Tuple[str, str]] = []
        self.vocab:         List[str]             = []
        self.normalization: str                   = "standard"

    def fit(self, corpus_path: str, k: int, normalization: str = "standard") -> None:
        """
        Train BPE for k merge steps on the corpus at corpus_path.
        Stop early if no pair appears at least twice (nothing useful to merge).
        """
        self.normalization = normalization
        words = words_from_file_norm(corpus_path, normalization=normalization)
        seqs  = corpus_to_symbol_sequences(words)

        for step in range(k):
            pair_counts = get_pair_counts(seqs)
            if not pair_counts:
                break
            (a, b), cnt = pair_counts.most_common(1)[0]
            if cnt < 2:
                break
            seqs = apply_merge(seqs, a, b)
            self.merges.append((a, b))

        # Vocabulary = all distinct symbols remaining after merges
        vocab_set = set()
        for s in seqs:
            vocab_set.update(s)
        # Sort: short first, then lex — deterministic ordering
        self.vocab = sorted(vocab_set, key=lambda t: (len(t), t))

    def tokenize_word(self, word: str) -> List[str]:
        """Apply learned merges to a single word."""
        symbols = list(word_to_symbols(word))
        for a, b in self.merges:
            i = 0
            while i < len(symbols) - 1:
                if symbols[i] == a and symbols[i + 1] == b:
                    symbols[i] = a + b
                    del symbols[i + 1]
                else:
                    i += 1
        return symbols

    def tokenize(self, text: str) -> List[str]:
        """Tokenize a full text using the learned merge rules."""
        words  = words_from_text_norm(text, self.normalization)
        tokens = []
        for w in words:
            tokens.extend(self.tokenize_word(w))
        return tokens

    def save_merges(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for a, b in self.merges:
                f.write(f"{a} {b}\n")

    @classmethod
    def load_merges(cls, path: str, normalization: str = "standard") -> "BPETokenizer":
        tok = cls()
        tok.normalization = normalization
        with open(path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    tok.merges.append((parts[0], parts[1]))
        return tok


# ── Evaluation metrics ───────────────────────────────────────────────────────
def compute_tokenization_metrics(
    text: str,
    merges: List[Tuple[str, str]],
    normalization: str = "standard",
) -> Dict[str, float]:
    """
    avg_tokens_per_word: compression ratio (lower = more merges used)
    word_as_token_rate:  fraction of words represented as a single token
    merge_use_rate:      fraction of learned merges that actually appear in text
    """
    tok = BPETokenizer()
    tok.merges        = merges
    tok.normalization = normalization

    words   = words_from_text_norm(text, normalization)
    if not words:
        return {}

    tokenized   = [tok.tokenize_word(w) for w in words]
    total_toks  = sum(len(t) for t in tokenized)
    single_tok  = sum(1 for t in tokenized if len(t) == 1)

    # Which merges are actually used?
    used = set()
    for toks in tokenized:
        for t in toks:
            for a, b in merges:
                if t == a + b:
                    used.add((a, b))

    return {
        "avg_tokens_per_word": round(total_toks / len(words), 4),
        "word_as_token_rate":  round(single_tok  / len(words), 4),
        "merge_use_rate":      round(len(used)   / max(len(merges), 1), 4),
    }
