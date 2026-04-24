# MIA — NLP II — Practical Assignments

Repository for the NLP II course practical work at the
[Maestría en Inteligencia Artificial (MIA)](https://github.com/FIUBA-Posgrado-Inteligencia-Artificial/CEIA-LLMIAG).

## Branch layout

| Branch | Assignment | Description |
|--------|-----------|-------------|
| `main` | TP-1 | TinyGPT — Decoding strategies and Mixture of Experts |
| `TP-2` | TP-2 | CV RAG Chatbot — Single-CV retrieval-augmented generation |
| `TP-3` | TP-3 | Multi-Agent CV RAG — Per-person CV agents with query routing |

---

## TP-1 — TinyGPT

A character-level GPT model trained on the Tiny Shakespeare dataset, extended
with advanced decoding strategies and a Mixture of Experts (MoE) architecture.

### Files

```text
TinyGPT.ipynb   # Main notebook with model, training, and experiments
trainer.py      # Training utilities (provided by the course)
```

### Task I — Decoding strategies

The original `generate` function was extended into `generateV2` supporting:

| Strategy | Behaviour |
|----------|-----------|
| **Greedy** (`temperature=0`) | Picks the highest-probability token at each step (deterministic). |
| **Temperature sampling** (`0 < temperature`) | Scales logits before sampling; lower values sharpen the distribution, higher values increase diversity. |
| **Top-k sampling** (`top_k > 0`) | Keeps only the k most probable tokens, zeroing out the rest before sampling. |
| **Top-p / nucleus sampling** (`top_p > 0`) | Keeps the smallest set of tokens whose cumulative probability exceeds p. |

These strategies can be combined (e.g., `temperature=0.8, top_k=50, top_p=0.9`).

### Task II — Mixture of Experts (MoE)

The dense FeedForward layer was replaced with a MoE layer composed of:

- **`Expert`** — A standard 2-layer MLP (expand 4x, ReLU, project back).
- **`Gate`** — A linear projection that produces per-token routing scores over all experts.
- **`MoELayer`** — Routes each token to the top-k experts (weighted by softmax gating scores) and combines their outputs.
- **`MoEFFN`** — Drop-in replacement for `FeedForward` that wraps `MoELayer`.

Configuration used: **4 experts, 1 active expert per token**.

### Conclusions

- **Decoding**: Greedy is deterministic but repetitive. Temperature provides a smooth creativity knob. Top-k and top-p truncate the distribution tail, preventing low-probability tokens while preserving diversity. Combining top-k + top-p with moderate temperature (~0.8) yields the best coherence/variety balance.
- **MoE**: Increases total parameters without proportionally increasing per-token compute. The gating network learns to specialize experts on different patterns, matching or exceeding dense model quality at lower inference cost.

### How to run

Open `TinyGPT.ipynb` in Google Colab (GPU runtime recommended) or locally with
Jupyter and run all cells sequentially. The `trainer.py` file must be in the same
directory.
