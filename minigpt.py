"""
Task 4 — MiniGPT: Decoder-Only Transformer
Manually implemented causal self-attention + standard PyTorch components.

Architecture choices vs. LLM theory:
  LayerNorm (not RMSNorm): RMSNorm omits mean subtraction → faster, but
    LayerNorm is simpler and correct for a from-scratch implementation.
  Learned positional embeddings (not RoPE): absolute positions work well
    for fixed-length contexts; RoPE is needed for length generalisation.
  Full MHA (not GQA): GQA reduces KV memory at inference, but at this
    scale MHA is clearer to implement and debug.
  Pre-LN: x = x + Attn(LN(x)) — residual path stays clean, training stable.
  Weight tying: lm_head.weight = tok_emb.weight — parameter savings.

Best results (k=1600): PPL ≈ 5.36 (vs. bigram-backoff ≈ 45, neural-ngram ≈ 52).
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Causal Self-Attention ─────────────────────────────────────────────────────
class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention — implemented from scratch.

    Single QKV projection (d_model → 3×d_model), then split into heads.
    Causal mask: lower-triangular bool buffer, registered as non-parameter.
    Scores = QK^T / sqrt(d_h) → mask(-inf) → softmax → dropout → ×V.

    Why masked_fill with dtype.min instead of -inf?
    → bfloat16 clips to -inf anyway; using finfo.min avoids NaN in softmax
      when the entire row is masked (no valid context at all).
    """

    def __init__(
        self,
        d_model:     int,
        n_head:      int,
        attn_pdrop:  float,
        resid_pdrop: float,
        block_size:  int,
    ):
        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        self.n_head = n_head
        self.dh     = d_model // n_head

        # Single projection for Q, K, V — efficient, one matmul
        self.qkv      = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out      = nn.Linear(d_model, d_model,     bias=True)
        self.attn_drop  = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)

        # Causal mask: pre-compute once, reuse across forward calls
        mask = torch.tril(torch.ones(block_size, block_size, dtype=torch.bool))
        self.register_buffer("mask", mask.view(1, 1, block_size, block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()

        # Project and split: each is (B, T, d_model) → reshape to heads
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.dh).transpose(1, 2)  # (B, h, T, dh)
        k = k.view(B, T, self.n_head, self.dh).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.dh).transpose(1, 2)

        # Scaled dot-product: scores = QK^T / sqrt(d_k)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)   # (B, h, T, T)

        # Causal mask: future positions → -inf → softmax(−inf) = 0
        causal = self.mask[:, :, :T, :T]
        scores = scores.masked_fill(~causal, torch.finfo(scores.dtype).min)

        # stable_softmax: subtract max before exp (overflow prevention)
        # Note: PyTorch's F.softmax does this internally — no manual step needed
        A = F.softmax(scores, dim=-1)
        A = self.attn_drop(A)

        # Weighted sum over values
        y = A @ v                                                    # (B, h, T, dh)
        y = y.transpose(1, 2).contiguous().view(B, T, C)            # (B, T, d_model)
        return self.resid_drop(self.out(y))


# ── Transformer Block (Pre-LN) ────────────────────────────────────────────────
class Block(nn.Module):
    """
    Pre-LayerNorm Transformer block.
    x = x + Attn(LN(x))    ← residual keeps gradient highway clean
    x = x + MLP(LN(x))     ← FFN: "bilgi deposu", mlp_ratio×d_model hidden
    GELU activation (smoother than ReLU, standard in modern LMs).
    """

    def __init__(
        self,
        d_model:     int,
        n_head:      int,
        mlp_ratio:   int   = 4,
        attn_pdrop:  float = 0.1,
        resid_pdrop: float = 0.1,
        block_size:  int   = 128,
    ):
        super().__init__()
        self.ln1  = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, attn_pdrop, resid_pdrop, block_size)
        self.ln2  = nn.LayerNorm(d_model)
        self.mlp  = nn.Sequential(
            nn.Linear(d_model, mlp_ratio * d_model),
            nn.GELU(),
            nn.Dropout(resid_pdrop),
            nn.Linear(mlp_ratio * d_model, d_model),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))   # residual (+x): gradyan otoyolu
        x = x + self.mlp(self.ln2(x))
        return x


# ── MiniGPT ───────────────────────────────────────────────────────────────────
class MiniGPT(nn.Module):
    """
    Decoder-only Transformer with weight tying and final LayerNorm.

    H^(0) = E_tok[idx] + E_pos[positions]   ← token + positional embeddings
    H^(l) = Block^(l)(H^(l-1))              ← L stacked blocks
    logits = LN(H^(L)) × W_vocab            ← projection to vocabulary

    Weight tying: lm_head.weight = tok_emb.weight
    → Saves ~V×d params; works because the input and output embedding
      spaces represent the same token semantics.
    """

    def __init__(
        self,
        vocab_size:  int,
        block_size:  int,
        n_layer:     int,
        n_head:      int,
        n_embd:      int,
        embd_pdrop:  float = 0.1,
        attn_pdrop:  float = 0.1,
        resid_pdrop: float = 0.1,
        weight_tying: bool = True,
    ):
        super().__init__()
        self.block_size = block_size

        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)   # learned absolute positions
        self.drop    = nn.Dropout(embd_pdrop)

        self.blocks  = nn.ModuleList([
            Block(n_embd, n_head, mlp_ratio=4,
                  attn_pdrop=attn_pdrop, resid_pdrop=resid_pdrop,
                  block_size=block_size)
            for _ in range(n_layer)
        ])

        self.ln_f    = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        if weight_tying:
            self.lm_head.weight = self.tok_emb.weight

        n_params = sum(p.numel() for p in self.parameters())
        print(f"[MiniGPT] {n_params/1e6:.2f}M params | "
              f"V={vocab_size} d={n_embd} L={n_layer} H={n_head} T={block_size}")

    def forward(
        self,
        idx:     torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ):
        B, T = idx.size()
        assert T <= self.block_size, f"Sequence length {T} > block_size {self.block_size}"

        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)  # (1, T)
        x   = self.drop(self.tok_emb(idx) + self.pos_emb(pos))

        for block in self.blocks:
            x = block(x)

        x      = self.ln_f(x)
        logits = self.lm_head(x)   # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # cross_entropy: doğru token ne kadar düşük skor aldı?
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-1,
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx:            torch.Tensor,
        max_new_tokens: int,
        temperature:    float          = 1.0,
        do_sample:      bool           = False,
        top_k:          Optional[int]  = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation with temperature scaling.
        temperature < 1 → sharper distribution (more conservative)
        temperature > 1 → flatter distribution (more creative)
        top_k → nucleus-style truncation before sampling
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Crop to block_size if needed (no KV cache in this impl)
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self.forward(idx_cond)
            logits    = logits[:, -1, :]           # last position

            # Temperature: stable_softmax(logits / T)
            # logits - max(logits) overflow prevention happens inside F.softmax
            logits = logits / max(temperature, 1e-5)

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs    = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, 1) if do_sample else probs.argmax(dim=-1, keepdim=True)
            idx      = torch.cat([idx, next_tok], dim=1)

        return idx
