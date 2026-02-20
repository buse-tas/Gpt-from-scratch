"""
Task 3 — Neural N-gram Language Model
Embedding lookup → concatenate context → MLP → logits

Design notes (from LLM theory):
- Neural n-gram bridges count-based n-grams and full Transformers.
- Embed each context token separately, then concatenate:
  input_dim = ctx_len × d_embed
- Optional hidden layer with GELU + Dropout (Task 3 Part 1).
- Optimizer ablation: Adam (lr=1e-3) > AdamW > SGD (Task 3 Part 2).
- Early stopping on validation PPL; best checkpoint reloaded for test.
- Best results: k=1600, n=3, d_embed=256, d_hidden=256, PPL ≈ 52.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional

BOS = "<bos>"
EOS = "<eos>"


# ── Dataset ───────────────────────────────────────────────────────────────────
class NeuroNGramDataset(Dataset):
    """
    (context_tokens, next_token) supervised pairs.
    Vocabulary built from TRAIN only — UNK for out-of-vocab tokens.
    """

    def __init__(self, pairs: List[Tuple[List[str], str]], stoi: Dict[str, int]):
        self.pairs = pairs
        self.stoi  = dict(stoi)
        if "<UNK>" not in self.stoi:
            self.stoi["<UNK>"] = len(self.stoi)
        self.unk_id = self.stoi["<UNK>"]

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        ctx_tokens, target = self.pairs[idx]
        x = torch.tensor([self.stoi.get(t, self.unk_id) for t in ctx_tokens], dtype=torch.long)
        y = torch.tensor(self.stoi.get(target, self.unk_id),                  dtype=torch.long)
        return x, y


# ── Model ─────────────────────────────────────────────────────────────────────
class NeuroNGram(nn.Module):
    """
    Architecture:
        Embedding(vocab, d_embed) × ctx_len
        → flatten → [Linear(in, d_hidden) → GELU → Dropout] → Linear(→ vocab)

    If d_hidden == 0: single linear layer (logistic regression over concat embeddings).
    This serves as the neural baseline before introducing position-aware attention.
    """

    def __init__(
        self,
        vocab_size: int,
        d_embed:    int,
        ctx_len:    int,
        d_hidden:   int   = 0,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.ctx_len = ctx_len
        self.emb     = nn.Embedding(vocab_size, d_embed)
        in_dim       = d_embed * max(1, ctx_len)

        if d_hidden > 0:
            self.ff = nn.Sequential(
                nn.Linear(in_dim, d_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_hidden, vocab_size),
            )
        else:
            self.ff = nn.Linear(in_dim, vocab_size)

    def forward(self, x_ctx: torch.Tensor) -> torch.Tensor:
        """x_ctx: (B, ctx_len) → logits: (B, vocab_size)"""
        emb = self.emb(x_ctx)               # (B, ctx_len, d_embed)
        emb = emb.reshape(emb.size(0), -1)  # (B, ctx_len * d_embed)
        return self.ff(emb)                  # (B, vocab_size)


# ── Vocabulary helpers ────────────────────────────────────────────────────────
def build_vocab(token_lines: List[List[str]]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Build vocabulary from TRAIN lines only.
    Special tokens in fixed positions: <UNK>=0, <PAD>=1, <bos>=2.
    Any token in valid/test but not in train maps to UNK.
    """
    special = ["<UNK>", "<PAD>", BOS]
    vocab   = set()
    for line in token_lines:
        vocab.update(line)
    all_tokens = special + sorted(vocab - set(special))
    stoi = {tok: i for i, tok in enumerate(all_tokens)}
    itos = {i: tok for tok, i in stoi.items()}
    return stoi, itos


def build_supervised_pairs(
    token_lines: List[List[str]],
    n:           int,
) -> List[Tuple[List[str], str]]:
    """
    Build (context[n-1], next_token) pairs with BOS left-padding.
    Context is always exactly (n-1) tokens; pad with BOS at line start.
    """
    pairs = []
    for line in token_lines:
        padded = [BOS] * (n - 1) + line
        for i in range(n - 1, len(padded)):
            ctx    = padded[i - (n - 1) : i]
            target = padded[i]
            pairs.append((ctx, target))
    return pairs


# ── Training loop ─────────────────────────────────────────────────────────────
def run_epoch(
    model:     NeuroNGram,
    loader:    DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device:    str,
) -> float:
    """
    One epoch of training or evaluation.
    Returns average cross-entropy loss.
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total, n = 0.0, 0

    with torch.set_grad_enabled(is_train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss   = F.cross_entropy(logits, y)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total += loss.item()
            n     += 1

    return total / max(n, 1)


def eval_ppl(model: NeuroNGram, loader: DataLoader, device: str) -> float:
    """PPL = exp(CE). Lower is better."""
    ce = run_epoch(model, loader, optimizer=None, device=device)
    return math.exp(min(ce, 50.0))


def train_with_early_stopping(
    model:      NeuroNGram,
    tr_loader:  DataLoader,
    va_loader:  DataLoader,
    max_epochs: int   = 20,
    patience:   int   = 3,
    lr:         float = 1e-3,
    device:     str   = "cpu",
    opt_name:   str   = "adam",     # "adam" | "adamw" | "sgd"
) -> dict:
    """
    Task 3 Part 2: optimizer ablation.
    Adam (lr=1e-3) consistently best across k ∈ {1600, 1800, 2000}.
    AdamW slightly worse; SGD substantially worse in this LR range.
    """
    if opt_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    elif opt_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")

    model.to(device)
    best_val_ppl  = float("inf")
    best_state    = None
    no_improve    = 0
    history       = {"train_ppl": [], "val_ppl": []}

    for epoch in range(max_epochs):
        tr_ce   = run_epoch(model, tr_loader, optimizer, device)
        va_ppl  = eval_ppl(model, va_loader, device)
        tr_ppl  = math.exp(min(tr_ce, 50.0))

        history["train_ppl"].append(tr_ppl)
        history["val_ppl"].append(va_ppl)
        print(f"  Epoch {epoch+1:2d} | train_ppl={tr_ppl:.2f} | val_ppl={va_ppl:.2f}")

        if va_ppl < best_val_ppl:
            best_val_ppl = va_ppl
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve   = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch+1} (patience={patience})")
                break

    if best_state:
        model.load_state_dict(best_state)
    return history
