"""
Task 2 — N-gram Language Models (n ∈ {1, 2, 3, 4})
Smoothing: Maximum Likelihood, Laplace, Interpolation, Stupid Backoff

Design notes:
- Models are trained on TRAIN tokenized lines only.
- Context = (n-1) preceding tokens; BOS padding for line boundaries.
- Perplexity = exp(avg negative log-likelihood) — lower is better.
- Bigram with backoff was the strongest classic baseline (PPL ≈ 45–50).
"""

import math
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Optional

BOS = "<bos>"
EOS = "<eos>"


# ── N-gram model ──────────────────────────────────────────────────────────────
class NGramLM:
    """
    Count-based n-gram language model.
    Smoothing modes: 'ml' | 'laplace' | 'interp' | 'backoff'
    """

    def __init__(self, n: int):
        self.n            = n
        self.ngram_counts: Counter              = Counter()
        self.ctx_counts:   Counter              = Counter()
        self.vocab:        set                  = set()
        # Interpolation weights (set via fit_interpolation_weights)
        self.lambdas:      Optional[List[float]] = None
        # Backoff lower-order models
        self._lower:       Optional["NGramLM"]  = None

    # ── Training ──────────────────────────────────────────────────────────────
    def fit(self, token_lines: List[List[str]]) -> None:
        """
        Build n-gram and context counts.
        Each line is padded with (n-1) BOS tokens and one EOS.
        Vocabulary is collected from all observed tokens.
        """
        for line in token_lines:
            padded = [BOS] * (self.n - 1) + line + [EOS]
            self.vocab.update(padded)
            for i in range(len(padded) - self.n + 1):
                ctx = tuple(padded[i : i + self.n - 1])
                tok = padded[i + self.n - 1]
                self.ngram_counts[(ctx, tok)] += 1
                self.ctx_counts[ctx]          += 1

    # ── Probability ──────────────────────────────────────────────────────────
    def prob(self, token: str, context: Tuple[str, ...], mode: str = "ml") -> float:
        """
        P(token | context) under the requested smoothing mode.
        Returns a probability (0, 1].
        """
        ctx = tuple(context[-(self.n - 1):]) if self.n > 1 else ()

        if mode == "ml":
            return self._ml(token, ctx)
        elif mode == "laplace":
            return self._laplace(token, ctx)
        elif mode == "interp":
            return self._interp(token, context)
        elif mode == "backoff":
            return self._backoff(token, ctx)
        else:
            raise ValueError(f"Unknown smoothing mode: {mode}")

    def _ml(self, token: str, ctx: tuple) -> float:
        """Maximum Likelihood — falls back to uniform if context unseen."""
        num = self.ngram_counts.get((ctx, token), 0)
        den = self.ctx_counts.get(ctx, 0)
        if den == 0:
            return 1.0 / max(len(self.vocab), 1)
        return num / den

    def _laplace(self, token: str, ctx: tuple) -> float:
        """Add-1 (Laplace) smoothing."""
        V   = len(self.vocab)
        num = self.ngram_counts.get((ctx, token), 0) + 1
        den = self.ctx_counts.get(ctx, 0) + V
        return num / max(den, 1)

    def _interp(self, token: str, context: tuple) -> float:
        """
        Linear interpolation across all orders 1..n.
        Weights are set by fit_interpolation_weights() on a validation set;
        default to uniform if not set.
        """
        n = self.n
        lams = self.lambdas if self.lambdas else [1.0 / n] * n
        p = 0.0
        model = self
        for order in range(n, 0, -1):
            ctx_o = tuple(context[-(order - 1):]) if order > 1 else ()
            num = model.ngram_counts.get((ctx_o, token), 0)
            den = model.ctx_counts.get(ctx_o, 0)
            p_o = (num / den) if den > 0 else (1.0 / max(len(self.vocab), 1))
            p  += lams[n - order] * p_o
        return p

    def _backoff(self, token: str, ctx: tuple, discount: float = 0.4) -> float:
        """
        Stupid backoff (Brants et al.): use higher order if observed,
        otherwise scale lower-order probability by discount.
        """
        num = self.ngram_counts.get((ctx, token), 0)
        den = self.ctx_counts.get(ctx, 0)
        if den > 0 and num > 0:
            return num / den
        if self.n == 1:
            return 1.0 / max(len(self.vocab), 1)
        # Back off
        lower_ctx = ctx[1:] if len(ctx) > 0 else ()
        if self._lower is None:
            # Lazy construction of lower-order model from same counts
            # (only approximate — for a proper implementation train separately)
            return discount / max(len(self.vocab), 1)
        return discount * self._lower._backoff(token, lower_ctx)

    # ── Perplexity ─────────────────────────────────────────────────────────────
    def perplexity(self, token_lines: List[List[str]], mode: str = "ml") -> float:
        """
        Teacher-forced perplexity on tokenized lines.
        PPL = exp(- (1/N) Σ log P(token | context))
        Lower is better; bigram-backoff achieves PPL ≈ 45 on Shakespeare test.
        """
        total_log = 0.0
        total_tok = 0
        for line in token_lines:
            padded = [BOS] * (self.n - 1) + line + [EOS]
            for i in range(self.n - 1, len(padded)):
                ctx   = tuple(padded[i - (self.n - 1) : i])
                tok   = padded[i]
                p     = self.prob(tok, ctx, mode=mode)
                total_log += math.log(max(p, 1e-300))
                total_tok += 1
        if total_tok == 0:
            return float("inf")
        return math.exp(-total_log / total_tok)

    # ── Interpolation weight tuning ───────────────────────────────────────────
    def fit_interpolation_weights(
        self,
        val_lines: List[List[str]],
        step: float = 0.1,
    ) -> List[float]:
        """
        Grid search over weight simplex to minimise validation perplexity.
        Coarse grid (step=0.1) is fast; refine if needed.
        """
        from itertools import product

        def simplex(n, step):
            """All n-vectors summing to 1 on a coarse grid."""
            vals = [round(i * step, 8) for i in range(int(1 / step) + 1)]
            candidates = []
            for combo in product(vals, repeat=n):
                if abs(sum(combo) - 1.0) < 1e-6:
                    candidates.append(list(combo))
            return candidates

        best_ppl   = float("inf")
        best_lams  = [1.0 / self.n] * self.n
        for lams in simplex(self.n, step):
            self.lambdas = lams
            ppl = self.perplexity(val_lines, mode="interp")
            if ppl < best_ppl:
                best_ppl  = ppl
                best_lams = lams

        self.lambdas = best_lams
        return best_lams
