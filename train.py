"""
Task 4 — Training & Evaluation Helpers
Minimal PyTorch loop with CUDA AMP (bfloat16) for A100-class GPUs.
Early stopping on validation perplexity.
"""

import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional


# ── Dataset ───────────────────────────────────────────────────────────────────
class GPTDataset(Dataset):
    """
    Transform a 1D token ID stream into overlapping (x, y) next-token pairs.
    x[i] = ids[i : i+T]
    y[i] = ids[i+1 : i+T+1]  ← teacher-forced targets, shifted by one
    """

    def __init__(self, ids_1d: np.ndarray, block_size: int):
        self.ids = torch.from_numpy(ids_1d.astype(np.int64))
        self.T   = int(block_size)

    def __len__(self):
        return max(0, self.ids.numel() - self.T)

    def __getitem__(self, i):
        return self.ids[i : i + self.T], self.ids[i + 1 : i + 1 + self.T]


def build_loaders(
    train_ids:   np.ndarray,
    val_ids:     np.ndarray,
    block_size:  int,
    batch_size:  int,
    num_workers: int  = 0,
    pin_memory:  bool = False,
) -> Tuple[DataLoader, DataLoader]:
    tr = GPTDataset(train_ids, block_size)
    va = GPTDataset(val_ids,   block_size)
    tr_loader = DataLoader(tr, batch_size=batch_size, shuffle=True,
                           num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
    va_loader = DataLoader(va, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
    print(f"[data] block={block_size} batch={batch_size} | "
          f"train_steps={len(tr_loader)} val_steps={len(va_loader)}")
    return tr_loader, va_loader


# ── Training epoch ─────────────────────────────────────────────────────────────
def train_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    device:    str,
    grad_clip: float = 1.0,
) -> float:
    """
    One training epoch with CUDA AMP (bfloat16).
    GradScaler handles mixed-precision backward; grad clipping prevents explosion.
    Returns average cross-entropy loss.
    """
    model.train()
    use_amp = device.startswith("cuda")
    scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)
    total, n = 0.0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
            _, loss = model(x, y)

        scaler.scale(loss).backward()

        if grad_clip is not None:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

        total += float(loss.item())
        n     += 1

    return total / max(n, 1)


# ── Evaluation ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def eval_ce(model: nn.Module, loader: DataLoader, device: str) -> float:
    """Teacher-forcing cross-entropy — no gradient, no AMP needed."""
    model.eval()
    total, n = 0.0, 0
    for x, y in loader:
        _, loss = model(x.to(device), y.to(device))
        total  += float(loss.item())
        n      += 1
    return total / max(n, 1)


def ppl_from_ce(ce: float) -> float:
    """PPL = exp(CE). Clip CE at 50 to avoid overflow."""
    return float(math.exp(min(ce, 50.0)))


# ── Full training run with early stopping ─────────────────────────────────────
def train_model(
    model:       nn.Module,
    tr_loader:   DataLoader,
    va_loader:   DataLoader,
    max_epochs:  int   = 30,
    patience:    int   = 4,
    lr:          float = 1e-4,
    device:      str   = "cpu",
    ckpt_path:   str   = "best_model.pt",
) -> dict:
    """
    Training loop matching Task 4 experiment setup.
    Best config: dropout=0.2, lr=1e-4 → test PPL ≈ 5.36 at k=1600.

    FLOPs estimate: ~6 × N × D
    (N = model parameters, D = total training tokens)
    """
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)

    best_val_ppl = float("inf")
    no_improve   = 0
    history      = {"train_ppl": [], "val_ppl": [], "epoch": []}

    for epoch in range(1, max_epochs + 1):
        tr_ce   = train_epoch(model, tr_loader, optimizer, device)
        va_ce   = eval_ce(model, va_loader, device)
        tr_ppl  = ppl_from_ce(tr_ce)
        va_ppl  = ppl_from_ce(va_ce)

        history["epoch"].append(epoch)
        history["train_ppl"].append(round(tr_ppl, 3))
        history["val_ppl"].append(round(va_ppl, 3))
        print(f"Epoch {epoch:3d} | train_ppl={tr_ppl:7.3f} | val_ppl={va_ppl:7.3f}")

        if va_ppl < best_val_ppl:
            best_val_ppl = va_ppl
            torch.save(model.state_dict(), ckpt_path)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (patience={patience})")
                break

    # Reload best checkpoint
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"Best val PPL: {best_val_ppl:.3f}")
    return history


def best_device(user_device: Optional[str] = None) -> str:
    """Auto-select: user preference > CUDA > MPS > CPU."""
    if user_device:
        return user_device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
