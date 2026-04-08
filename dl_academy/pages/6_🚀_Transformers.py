import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from utils.styles import apply_styles, formula, tip, section
from utils.visualizations import plot_attention_heatmap
from utils.quiz_engine import render_quiz, init_progress

st.set_page_config(page_title="Transformers", page_icon="🚀", layout="wide")
apply_styles()
init_progress()

section("🚀 Transformers & Attention",
        "The architecture behind GPT, BERT, and all modern AI — attention is all you need", "advanced")

tabs = st.tabs(["👁️ Attention", "🔑 Q-K-V", "🏗️ Architecture", "🤖 BERT & GPT", "🎨 Interactive", "✅ Quiz"])


# ══════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown("### The Attention Mechanism")
    st.markdown("""
**Attention** allows the model to focus on relevant parts of the input when processing each element.
Instead of compressing all context into a fixed hidden state, attention can **directly access** any previous position.
""")

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.markdown("#### Why Attention? The Problem with RNNs")
        st.markdown("""
In an RNN, the encoder must compress **the entire input** into a single hidden vector.

For long sentences this is impossible — information bottleneck!

**Bahdanau Attention** (2014) solution:
- Keep **all encoder hidden states**
- Learn to **attend** to relevant ones for each output step
- Compute a context vector as a **weighted sum** of all hidden states

This was the precursor to the full Transformer.
        """)
        formula("context_t = Σᵢ αᵢ · hᵢ")
        formula("αᵢ = softmax(score(sₜ, hᵢ))")

    with col2:
        st.markdown("#### 🎨 Attention Heatmap")
        tokens = ["I", "love", "deep", "learning"]
        fig = plot_attention_heatmap(tokens)
        st.pyplot(fig)
        st.markdown("""
Each row = a query position  
Each column = a key position  
Darker = more attention paid
        """)

    st.markdown("---")
    st.markdown("#### Self-Attention: Every Position Attends to All Others")
    st.markdown("""
In **self-attention**, each position in the sequence attends to **all other positions** in the same sequence.
This allows the model to capture long-range dependencies without sequential processing!

Example: "The animal didn't cross the street because **it** was too tired"
- The model can directly learn that "it" refers to "animal" — no sequential bottleneck!
    """)

    tip("Attention is O(n²) in sequence length — each position attends to all others. This is Transformers' main scaling challenge for long sequences.")


# ══════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("### Query, Key, Value — The Attention Formula")
    st.markdown("""
Scaled dot-product attention reformulates the problem as a **database lookup**:
- **Query (Q)**: What am I looking for?
- **Key (K)**: What does each position advertise?
- **Value (V)**: What is the actual content at each position?
""")

    formula("Attention(Q, K, V) = softmax(QKᵀ / √dₖ) · V")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
**Step by step:**

1. Compute Q, K, V by multiplying input X by learned matrices Wq, Wk, Wv
2. Compute attention scores: QKᵀ (dot product of every query with every key)
3. Scale by √dₖ to prevent softmax saturation
4. Apply softmax → attention weights (sum to 1)
5. Weighted sum of Values: output = weights × V
        """)

    with col2:
        st.markdown("**Why scale by √dₖ?**")
        st.markdown("""
For large dₖ, dot products grow large → softmax becomes peaky (extreme) → gradients vanish.
Scaling by √dₖ keeps magnitudes stable.

**Multi-Head Attention:**
Run attention **h times** in parallel with different learned Q,K,V projections,
then concatenate and project:
        """)
        formula("MultiHead(Q,K,V) = Concat(head₁,...,headₕ) · Wₒ")
        st.markdown("Each head can attend to different aspects of the input!")

    st.markdown("---")
    st.markdown("#### 🎨 Attention Score Visualization")
    np.random.seed(42)
    tokens = ["The", "cat", "sat", "on", "the", "mat"]
    n = len(tokens)
    d_k = 8

    Q = np.random.randn(n, d_k)
    K = np.random.randn(n, d_k)
    scores = Q @ K.T / np.sqrt(d_k)
    weights = np.exp(scores) / np.exp(scores).sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor("#1E2130")
    for ax, data, title, cmap in zip(axes,
                                      [scores, weights],
                                      ["Raw Scores (QKᵀ/√dₖ)", "Attention Weights (after softmax)"],
                                      ["RdBu_r", "YlOrRd"]):
        ax.set_facecolor("#1E2130")
        im = ax.imshow(data, cmap=cmap, aspect="auto")
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(tokens, color="#E8EAF0", rotation=30)
        ax.set_yticklabels(tokens, color="#E8EAF0")
        ax.set_title(title, color="#E8EAF0", fontsize=11)
        plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    st.pyplot(fig)


# ══════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("### Transformer Architecture")
    st.markdown("""
The full Transformer (Vaswani et al., 2017) stacks encoder and decoder blocks.
The key insight: **attention replaces recurrence entirely** — enabling massive parallelization!
""")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Encoder Block")
        components = [
            ("Input Embedding + Positional Encoding", "#6C63FF"),
            ("Multi-Head Self-Attention", "#FF6584"),
            ("Add & Norm (Residual)", "#43D9AD"),
            ("Feed-Forward Network", "#FF6584"),
            ("Add & Norm (Residual)", "#43D9AD"),
        ]
        for comp, color in components:
            st.markdown(
                f'<div style="background:{color}15;border:1px solid {color}44;border-radius:6px;'
                f'padding:8px 14px;margin:4px 0;color:#E8EAF0;font-size:13px">'
                f'<span style="color:{color}">▶</span> {comp}</div>',
                unsafe_allow_html=True,
            )

    with col2:
        st.markdown("#### Positional Encoding")
        formula("PE(pos,2i) = sin(pos/10000^(2i/d))")
        formula("PE(pos,2i+1) = cos(pos/10000^(2i/d))")
        st.markdown("""
Since Transformers process all positions in parallel, they have **no inherent sense of order**.
Positional encodings inject position information as a pattern of sine/cosine waves.

This allows the model to learn:
- Absolute position (where am I?)
- Relative position (how far apart are these two words?)
        """)

        # Plot positional encoding
        d_model = 16
        max_len = 20
        pe = np.zeros((max_len, d_model))
        for pos in range(max_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = np.sin(pos / 10000 ** (2*i/d_model))
                if i+1 < d_model:
                    pe[pos, i+1] = np.cos(pos / 10000 ** (2*i/d_model))
        fig, ax = plt.subplots(figsize=(6, 3))
        fig.patch.set_facecolor("#1E2130"); ax.set_facecolor("#1E2130")
        im = ax.imshow(pe.T, cmap="RdBu_r", aspect="auto")
        ax.set_xlabel("Position", color="#E8EAF0"); ax.set_ylabel("Dimension", color="#E8EAF0")
        ax.set_title("Positional Encoding", color="#E8EAF0", fontsize=11)
        ax.tick_params(colors="#E8EAF0")
        plt.colorbar(im, ax=ax, fraction=0.046)
        st.pyplot(fig)

    st.markdown("---")
    st.markdown("#### Layer Normalization & Residual Connections")
    formula("output = LayerNorm(x + Sublayer(x))")
    st.markdown("""
Every sublayer (attention, FFN) uses a **residual connection** + **layer normalization**:
- **Residual**: gradient highway — same trick as ResNet, enables deep networks
- **LayerNorm**: normalizes across features (not batch) — stable for sequences of varying length
    """)


# ══════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("### BERT, GPT, and the Transformer Family")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### BERT — Encoder-only")
        st.markdown("""
**B**idirectional **E**ncoder **R**epresentations from **T**ransformers (Google, 2018)

- Uses **only the encoder** stack
- Trained with **Masked Language Modeling** (predict masked tokens)
- **Bidirectional**: attends to both left and right context
- Perfect for: classification, NER, question answering, text similarity

**Pretraining tasks:**
1. **MLM**: Mask 15% of tokens → predict them
2. **NSP**: Predict if sentence B follows sentence A
        """)

    with col2:
        st.markdown("#### GPT — Decoder-only")
        st.markdown("""
**G**enerative **P**re-trained **T**ransformer (OpenAI, 2018+)

- Uses **only the decoder** stack
- Trained with **causal language modeling** (predict next token)
- **Unidirectional**: only attends to previous tokens (causal masking)
- Perfect for: text generation, chat, code completion

GPT scaling:
- GPT-1: 117M params
- GPT-2: 1.5B params  
- GPT-3: 175B params
- GPT-4: ~1T params (estimated)
        """)

    st.markdown("---")
    st.markdown("#### 🏆 Transformer Variants Timeline")
    models = [
        ("2017", "Transformer", "Google", "Original seq2seq transformer", "#6C63FF"),
        ("2018", "BERT", "Google", "Bidirectional encoder, MLM pretraining", "#FF6584"),
        ("2018", "GPT-1", "OpenAI", "Autoregressive decoder, causal LM", "#43D9AD"),
        ("2019", "GPT-2", "OpenAI", "1.5B params, zero-shot generation", "#43D9AD"),
        ("2019", "RoBERTa", "Facebook", "Better BERT training", "#FF6584"),
        ("2020", "GPT-3", "OpenAI", "175B, few-shot learning", "#43D9AD"),
        ("2020", "ViT", "Google", "Vision Transformer — images as patches", "#FFB347"),
        ("2022", "ChatGPT", "OpenAI", "RLHF fine-tuned GPT-3.5", "#43D9AD"),
        ("2023", "GPT-4", "OpenAI", "Multimodal, frontier model", "#43D9AD"),
        ("2024", "Llama 3", "Meta", "Open-source frontier LLM", "#FFB347"),
    ]

    for year, model, org, desc, color in models:
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:12px;padding:8px 0;border-bottom:1px solid #3A3F5C">'
            f'<span style="color:#8890B5;min-width:40px;font-size:13px">{year}</span>'
            f'<span style="color:{color};font-weight:700;min-width:90px">{model}</span>'
            f'<span style="color:#FFB347;min-width:80px;font-size:12px">{org}</span>'
            f'<span style="color:#8890B5;font-size:13px">{desc}</span></div>',
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown("### 🎨 Interactive: Attention Heatmap Explorer")

    st.markdown("Enter a sentence and explore how self-attention might look:")
    sentence = st.text_input("Enter a sentence (space-separated words):",
                              value="The quick brown fox jumps over the lazy dog")
    words = sentence.split()[:12]  # cap at 12
    n = len(words)

    head = st.slider("Attention head (simulated)", 1, 8, 1)
    np.random.seed(head * 7 + 3)

    # Simulate different attention patterns for different heads
    patterns = {
        1: "local",   # attend to nearby words
        2: "subject", # attend to first word
        3: "random",
        4: "diagonal",# attend to self
        5: "last",    # attend to last word
        6: "random",
        7: "local",
        8: "diagonal",
    }
    pattern = patterns.get(head, "random")

    raw = np.random.randn(n, n)
    if pattern == "local":
        for i in range(n):
            for j in range(n):
                raw[i,j] += 3 * np.exp(-abs(i-j) * 0.7)
    elif pattern == "diagonal":
        raw += np.eye(n) * 3
    elif pattern == "subject":
        raw[:, 0] += 2
    elif pattern == "last":
        raw[:, -1] += 2

    attn = np.exp(raw) / np.exp(raw).sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(max(6, n*0.8), max(5, n*0.7)))
    fig.patch.set_facecolor("#1E2130"); ax.set_facecolor("#1E2130")
    im = ax.imshow(attn, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(words, color="#E8EAF0", rotation=45, ha="right")
    ax.set_yticklabels(words, color="#E8EAF0")
    ax.set_title(f"Simulated Attention Head {head} ({pattern} pattern)", color="#E8EAF0", fontsize=12)
    ax.set_ylabel("Query (attending FROM)", color="#E8EAF0")
    ax.set_xlabel("Key (attending TO)", color="#E8EAF0")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{attn[i,j]:.2f}", ha="center", va="center",
                    color="black" if attn[i,j] > 0.4 else "white", fontsize=8)
    plt.colorbar(im, ax=ax, label="Attention Weight")
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown(f"""
**Reading the heatmap:**
- Row = query position (word currently being processed)
- Column = key position (word being attended to)
- Brighter = more attention paid

**Head {head} pattern: `{pattern}`** — different heads learn different linguistic patterns!
    """)


# ══════════════════════════════════════════════════════════
with tabs[5]:
    render_quiz("transformer_quiz", [
        {
            "question": "What problem did attention mechanisms solve in RNN-based seq2seq models?",
            "options": [
                "Slow training speed",
                "The information bottleneck of compressing all context into one fixed-size hidden vector",
                "The need for labeled data",
                "Overfitting on long sequences"
            ],
            "answer": 1,
            "explanation": "RNN encoders compress the whole input into one vector. Attention keeps all hidden states and lets the decoder focus on relevant ones."
        },
        {
            "question": "In self-attention, what do Query, Key, and Value represent?",
            "options": [
                "Input, output, and hidden state of the network",
                "The three weight matrices for backpropagation",
                "What I'm looking for (Q), what others advertise (K), and their actual content (V)",
                "Query language, knowledge, and vocabulary"
            ],
            "answer": 2,
            "explanation": "Q asks 'what do I need?', K broadcasts 'what do I have?', V is the content retrieved. Attention score = Q·Kᵀ/√dₖ."
        },
        {
            "question": "Why does the Transformer need positional encoding?",
            "options": [
                "To speed up matrix multiplication",
                "Because attention processes all positions in parallel and has no inherent sense of order",
                "To reduce the number of parameters",
                "To handle images"
            ],
            "answer": 1,
            "explanation": "Unlike RNNs, attention sees all positions simultaneously. Positional encodings inject position info using sin/cos patterns."
        },
        {
            "question": "What makes BERT different from GPT in its attention pattern?",
            "options": [
                "BERT uses RNN layers; GPT uses attention",
                "BERT is bidirectional (attends to left and right context); GPT is causal (left-only)",
                "BERT is smaller than GPT",
                "GPT is trained on more data"
            ],
            "answer": 1,
            "explanation": "BERT: encoder-only, bidirectional → great for understanding tasks. GPT: decoder-only, causal masking → great for generation."
        },
        {
            "question": "Multi-head attention runs attention h times in parallel because:",
            "options": [
                "It's faster to parallelize",
                "Different heads learn different types of relationships (syntactic, semantic, local, global)",
                "It prevents overfitting",
                "The single head approach was patented"
            ],
            "answer": 1,
            "explanation": "Each head can specialize: one attends to nearby words, another to subject-verb agreement, another to coreference, etc."
        },
    ])
