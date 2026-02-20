# From N-grams to Transformers: Language Modeling on Shakespeare

Building GPT from Scratch (SoSe 2025) — Final Project  
Instructor: Prof. Dr. Elia Bruni | Student: Buse Taş (1006526) / Group 37  
Universität Osnabrück — Institute of Cognitive Science

## Table of Contents
- [Overview](#overview)
- [QuickStart](#quickstart)
- [Structure of this Repository](#structure-of-this-repository)
- [Task 1: BPE Tokenizer](#task-1-bpe-tokenizer)
- [Task 2: N-gram Language Models](#task-2-n-gram-language-models)
- [Task 3: Neural N-gram Models](#task-3-neural-n-gram-models)
- [Task 4: MiniGPT Transformer](#task-4-minigpt-transformer)
- [Final Comparison](#final-comparison)
- [References](#references)

## Overview

This project traces the full evolutionary arc of language modeling by implementing four model families from scratch on the Shakespeare corpus, using BPE tokenization throughout for a consistent comparison. Starting from classical count-based n-grams and ending with a decoder-only Transformer, each stage is a strict extension of the previous one: the tokenizer feeds into the n-gram models, which establish a baseline for the neural n-gram, which in turn sets the context for understanding what attention and positional encoding add.

All models are trained on a TRAIN split, tuned on a VALIDATION split, and reported on a held-out TEST split. The BPE tokenizer is trained on TRAIN only; the same frozen merge rules are then applied to every split. Vocabulary for all neural models is built from TRAIN only; tokens appearing in validation or test but not in training are mapped to UNK. This discipline prevents any form of leakage across the pipeline.

The Transformer (MiniGPT) achieves test perplexity ≈ 5.36 at BPE vocabulary size k=1600, compared to ≈ 45 for the best n-gram (bigram with stupid backoff) and ≈ 52 for the best neural n-gram. This factor-of-eight improvement demonstrates concretely what attention over the full context window provides that fixed-window n-gram approaches cannot.

## QuickStart

```bash
git clone https://github.com/YOUR_USERNAME/gpt-from-scratch
cd gpt-from-scratch
python -m venv venv
source venv/bin/activate        # Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

Place the Shakespeare corpus files in `data/`:
```
data/
  Shakespeare_clean_train.txt
  Shakespeare_clean_valid.txt
  Shakespeare_clean_test.txt
```

Run the full pipeline from the notebook:
```bash
jupyter notebook BuseTas_Final_Notebook_Group37.ipynb
```

Or run individual tasks:
```python
# Task 1: Train BPE
from src.tokenizer.bpe import BPETokenizer
tok = BPETokenizer()
tok.fit("data/Shakespeare_clean_train.txt", k=1600, normalization="standard")
tok.save_merges("artifacts/bpe_merges_k1600.txt")

# Task 4: Train MiniGPT
from src.models.minigpt import MiniGPT
from src.train import build_loaders, train_model, best_device

device = best_device()
model  = MiniGPT(vocab_size=8000, block_size=128, n_layer=4, n_head=4, n_embd=128,
                 embd_pdrop=0.2, attn_pdrop=0.2, resid_pdrop=0.2)
# ... build loaders and call train_model(model, tr_loader, va_loader, device=device)
```

## Structure of this Repository

```
.
├── BuseTas_Final_Notebook_Group37.ipynb   # Full experiment notebook
├── README.md
├── requirements.txt
├── data/                                  # Shakespeare train/valid/test splits
├── artifacts/                             # Saved merges, checkpoints, CSVs
└── src/
    ├── tokenizer/
    │   └── bpe.py              # BPETokenizer: fit, tokenize, save/load merges
    ├── models/
    │   ├── ngram.py            # NGramLM: ML, Laplace, Interpolation, Backoff
    │   ├── neural_ngram.py     # NeuroNGram: embedding concat MLP + training loop
    │   └── minigpt.py          # CausalSelfAttention, Block (Pre-LN), MiniGPT
    └── train.py                # GPTDataset, build_loaders, train_model (AMP), eval_ce
```

## Task 1: BPE Tokenizer

Byte Pair Encoding is a subword tokenization algorithm that iteratively merges the most frequent adjacent symbol pair in a corpus until a target vocabulary size k is reached. Words are first split into characters with a `</w>` end-of-word sentinel appended, so merges are constrained to stay within word boundaries. The sentinel also means the tokenizer can reconstruct word boundaries during decoding.

The tokenizer is trained on the TRAIN split only. The same frozen merge rules are then applied to TRAIN, VALID, and TEST for downstream evaluation. This matters: learning merges from the full corpus would let the tokenizer see test vocabulary, inflating downstream metrics.

The experiment sweeps k ∈ {1000, 1200, 1400, 1600, 1800, 2000}. Three metrics summarise behavior. Average tokens per word decreases from 1.50 (k=1000) to 1.29 (k=2000) — larger k compresses more aggressively. Word-as-single-token rate (fraction of words represented by exactly one token) increases with k. Merge use rate tells what fraction of the learned rules actually fire on test data. The gains slow down noticeably above k=1600, which is why k=1600 becomes the default for Tasks 3 and 4.

Two normalization schemes are compared: `standard` (lowercase + alphabetic sequences only) and `aggressive_clean` (keeps apostrophes, removes URLs). Standard normalization was used for the final results.

## Task 2: N-gram Language Models

N-gram models estimate the probability of the next token given the previous n-1 tokens. Four estimators are implemented and compared: maximum likelihood (ML), Laplace (add-one smoothing), linear interpolation, and stupid backoff.

ML assigns zero probability to unseen n-grams, which causes infinite perplexity on any test sentence containing an unseen context. Laplace adds one pseudo-count to every (context, token) pair, trading off some accuracy on seen events for finite probability everywhere. Interpolation mixes all orders from unigram to n-gram with learned weights tuned on the validation set. Stupid backoff uses the full-order count when available and recursively backs off to lower orders scaled by a fixed discount (0.4) otherwise — it is not a true probability distribution but works surprisingly well in practice.

Among all configurations, bigram (n=2) with stupid backoff achieves the lowest test perplexity (≈ 45–50 across k values). Higher order models overfit the training data without sufficient smoothing. The main limitation of all n-gram approaches is the fixed context window: a trigram can only look back two tokens regardless of how far the relevant context actually is.

## Task 3: Neural N-gram Models

The neural n-gram replaces the count table with a neural function: embed each of the n-1 context tokens, concatenate the embeddings into a single flat vector, and pass it through an MLP to produce logits over the vocabulary. This is fundamentally a fixed-window model like the n-gram, but it generalizes to unseen contexts through the shared embedding space.

The architecture sweeps embedding dimension (128 or 256), hidden dimension (0 = linear, or 256), context order n ∈ {2, 3}, and BPE size k. Training uses early stopping on validation perplexity with patience=3. Task 3 Part 2 adds an optimizer ablation: Adam (lr=1e-3) > AdamW > SGD with momentum in this setup. The best configuration (k=1600, n=3, d_embed=256, d_hidden=256, Adam) achieves test PPL ≈ 52, which is notably worse than the bigram backoff. The neural model needs to learn from fewer effective examples because the context window is wider, and cross-entropy training does not directly optimise for perplexity the way count-based methods implicitly do.

## Task 4: MiniGPT Transformer

MiniGPT is a decoder-only Transformer that replaces the fixed-window context of n-gram models with full-sequence causal attention. Every token can attend to every previous token in the sequence, weighted by learned query-key similarity.

The attention mechanism is implemented from scratch in `CausalSelfAttention`. A single linear projection maps the input to Q, K, V simultaneously (d_model → 3×d_model), which is then split and reshaped into heads. Scaled dot-product attention computes scores as QK^T / sqrt(d_h), applies a lower-triangular causal mask to set future positions to -inf before softmax, and weights the value vectors by the resulting attention distribution. The causal mask is pre-computed once as a buffer and reused at every forward pass.

Each Transformer block uses Pre-LayerNorm: the normalisation is applied to the input before the attention or FFN sublayer, not after. This keeps the residual stream unnormalised throughout the depth of the network, which stabilises gradients and makes the model easier to train without careful learning rate tuning.

The training loop uses CUDA Automatic Mixed Precision with bfloat16 for matrix multiplications, which halves memory bandwidth requirements on A100-class GPUs while keeping loss accumulation in FP32. A GradScaler manages the scaling of gradients to prevent underflow in reduced precision. Gradient clipping at 1.0 is applied every step.

**Best results:** dropout=0.2, lr=1e-4, k=1600 → test PPL = 5.36. PPL increases modestly as k grows (5.47 at k=1800, 5.50 at k=2000), consistent with the larger vocabulary being harder to predict.

Key design choices and their justifications:

| Component | This implementation | Modern LLMs (e.g. LLaMA 3) | Why the difference? |
|-----------|--------------------|-----------------------------|---------------------|
| Norm | LayerNorm | RMSNorm | RMSNorm omits mean → faster; LayerNorm is cleaner for a from-scratch study |
| Position | Learned absolute | RoPE | RoPE generalises better to lengths not seen at training time |
| Attention | Full MHA | GQA | GQA saves KV cache memory at inference; not critical at this scale |
| Activation | GELU | SwiGLU | SwiGLU is gated and more expressive; GELU is standard and simple |

## Final Comparison

| Model | Configuration | Test PPL |
|-------|---------------|----------|
| N-gram (bigram, backoff) | k=1600 | ≈ 45 |
| Neural N-gram | k=1600, n=3, d=256 | ≈ 52 |
| MiniGPT | k=1600, d=128, L=4, H=4 | **5.36** |

The Transformer's advantage is entirely attributable to its ability to use the full context window: when predicting the word "king", it can attend to any mention of the same character from fifty tokens ago, whereas a trigram can only see the two immediately preceding tokens.

## References

- Sennrich et al., *Neural Machine Translation of Rare Words with Subword Units (BPE)*, ACL 2016. https://arxiv.org/abs/1508.07909
- Kneser & Ney, *Improved backing-off for m-gram language modeling*, ICASSP 1995.
- Brants et al., *Large Language Models in Machine Translation (Stupid Backoff)*, EMNLP 2007.
- Bengio et al., *A Neural Probabilistic Language Model*, JMLR 2003.
- Vaswani et al., *Attention Is All You Need*, NeurIPS 2017. https://arxiv.org/abs/1706.03762
- Radford et al., *Language Models are Unsupervised Multitask Learners (GPT-2)*, OpenAI 2019.
