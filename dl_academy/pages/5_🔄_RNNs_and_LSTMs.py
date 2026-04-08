import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from utils.styles import apply_styles, formula, tip, section
from utils.quiz_engine import render_quiz, init_progress

st.set_page_config(page_title="RNNs & LSTMs", page_icon="🔄", layout="wide")
apply_styles()
init_progress()

section("🔄 RNNs & LSTMs",
        "Sequence modeling, memory, and the gates that made NLP possible", "intermediate")

tabs = st.tabs(["📜 RNNs", "🌊 Vanishing Gradient", "🚪 LSTM Gates", "🔀 GRU", "🧪 Sequence Demo", "✅ Quiz"])


# ══════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown("### Recurrent Neural Networks (RNNs)")
    st.markdown("""
Standard MLPs treat each input independently. But **sequences** have context — what came before matters.
**RNNs** maintain a hidden state that carries information across timesteps.
""")

    col1, col2 = st.columns([1.2, 1])
    with col1:
        # RNN unrolled diagram
        fig, ax = plt.subplots(figsize=(9, 4))
        fig.patch.set_facecolor("#1E2130"); ax.set_facecolor("#1E2130"); ax.axis("off")

        steps = 4
        labels_x = ["x₁", "x₂", "x₃", "xₜ"]
        labels_h = ["h₁", "h₂", "h₃", "hₜ"]
        labels_y = ["ŷ₁", "ŷ₂", "ŷ₃", "ŷₜ"]
        xs = np.linspace(0.1, 0.9, steps)

        for i, (x_pos, lx, lh, ly) in enumerate(zip(xs, labels_x, labels_h, labels_y)):
            # Input
            ax.text(x_pos, 0.1, lx, ha="center", va="center", color="#6C63FF",
                    fontsize=13, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#252A3D", edgecolor="#6C63FF"))
            # Hidden
            circle = plt.Circle((x_pos, 0.5), 0.07, color="#FF6584", zorder=5)
            ax.add_patch(circle)
            ax.text(x_pos, 0.5, lh, ha="center", va="center", color="white", fontsize=10, fontweight="bold", zorder=6)
            # Output
            ax.text(x_pos, 0.9, ly, ha="center", va="center", color="#43D9AD",
                    fontsize=11, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#252A3D", edgecolor="#43D9AD"))
            # Input → Hidden
            ax.annotate("", xy=(x_pos, 0.43), xytext=(x_pos, 0.17),
                        arrowprops=dict(arrowstyle="->", color="#8890B5", lw=1.5))
            # Hidden → Output
            ax.annotate("", xy=(x_pos, 0.83), xytext=(x_pos, 0.57),
                        arrowprops=dict(arrowstyle="->", color="#8890B5", lw=1.5))
            # Hidden → next hidden
            if i < steps - 1:
                ax.annotate("", xy=(xs[i+1] - 0.07, 0.5), xytext=(x_pos + 0.07, 0.5),
                            arrowprops=dict(arrowstyle="->", color="#FFB347", lw=2))

        ax.text(0.5, 0.5, "", ha="center")
        ax.text(-0.02, 0.5, "h₀→", color="#FFB347", va="center", fontsize=11)
        ax.set_xlim(-0.05, 1.0); ax.set_ylim(0, 1)
        ax.set_title("RNN Unrolled Through Time (orange = hidden state flow)", color="#E8EAF0", fontsize=12)
        st.pyplot(fig)

    with col2:
        st.markdown("#### The RNN Equations")
        formula("hₜ = tanh(Wₕ·hₜ₋₁ + Wₓ·xₜ + b)")
        formula("ŷₜ = softmax(Wᵧ·hₜ + bᵧ)")
        st.markdown("""
**hₜ** = hidden state at time t (the "memory")  
**xₜ** = input at time t  
**Wₕ** = recurrent weight matrix (shared across all t)  
**Wₓ** = input weight matrix  

**Key insight:** The **same weights** are used at every timestep — 
this is why RNNs can handle variable-length sequences!
        """)
        tip("Parameters are shared across time → RNNs have far fewer parameters than MLPs for sequence tasks!")

    st.markdown("---")
    st.markdown("#### Types of RNN Tasks")
    task_types = [
        ("One → One", "Standard MLP", "Image classification"),
        ("One → Many", "Single input, sequence output", "Image captioning"),
        ("Many → One", "Sequence input, single output", "Sentiment analysis"),
        ("Many → Many (sync)", "Sequence → same-length sequence", "Video frame labeling"),
        ("Many → Many (async)", "Encoder + Decoder", "Machine translation"),
    ]
    for rnn_type, desc, example in task_types:
        st.markdown(
            f'<div style="background:#1E2130;border-left:3px solid #6C63FF;border-radius:0 8px 8px 0;'
            f'padding:8px 14px;margin:4px 0;display:flex;gap:20px">'
            f'<span style="color:#6C63FF;font-weight:700;min-width:180px">{rnn_type}</span>'
            f'<span style="color:#8890B5;min-width:200px">{desc}</span>'
            f'<span style="color:#43D9AD">e.g. {example}</span></div>',
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("### The Vanishing Gradient Problem")
    st.markdown("""
RNNs struggle with **long sequences** because of the vanishing gradient problem.
""")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
**The Problem:**

During backpropagation through time (BPTT), gradients are multiplied at each timestep:

∂L/∂h₁ = ∂L/∂hₜ · (Wₕ)ᵗ⁻¹

If **|Wₕ| < 1**: gradients **vanish** exponentially → early timesteps don't learn  
If **|Wₕ| > 1**: gradients **explode** → unstable training

For a 100-step sequence, the gradient might be multiplied by 0.9 a hundred times:
**0.9¹⁰⁰ ≈ 0.000027** — essentially zero!
        """)
    with col2:
        steps = 50
        t = np.arange(steps)
        vanish = 0.9 ** t
        explode = np.minimum(1.1 ** t, 1e6)

        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        fig.patch.set_facecolor("#1E2130")
        for ax, data, label, color in zip(axes,
                                           [vanish, explode],
                                           ["Vanishing (×0.9 each step)", "Exploding (×1.1 each step)"],
                                           ["#FF6584", "#FFB347"]):
            ax.set_facecolor("#1E2130")
            ax.plot(t, data, color=color, linewidth=2.5)
            ax.fill_between(t, data, alpha=0.15, color=color)
            ax.set_title(label, color="#E8EAF0", fontsize=10)
            ax.set_xlabel("Timestep", color="#E8EAF0"); ax.set_ylabel("Gradient magnitude", color="#E8EAF0")
            ax.tick_params(colors="#E8EAF0"); ax.grid(True, color="#3A3F5C", alpha=0.4)
            for sp in ax.spines.values(): sp.set_edgecolor("#3A3F5C")
        plt.tight_layout()
        st.pyplot(fig)

    st.markdown("---")
    st.markdown("#### Solutions to Vanishing Gradients")
    solutions = [
        ("🚪 LSTM", "Learned gates control what to remember/forget — maintains long-term memory"),
        ("🔀 GRU", "Simplified LSTM with fewer gates — similar performance, faster training"),
        ("✂️ Gradient Clipping", "Cap gradient norm at a threshold — prevents explosion"),
        ("🔄 Truncated BPTT", "Only backprop through last k steps — approximate but faster"),
        ("🚀 Transformer", "Attention directly connects any two positions — no sequential bottleneck"),
    ]
    for icon_title, desc in solutions:
        st.markdown(
            f'<div class="concept-box" style="padding:12px 18px;margin:6px 0">'
            f'<span style="color:#6C63FF;font-weight:700">{icon_title}</span>'
            f'<span style="color:#8890B5;margin-left:16px">{desc}</span></div>',
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("### LSTM — Long Short-Term Memory")
    st.markdown("""
LSTMs solve the vanishing gradient problem by using **gates** to control information flow.
They maintain two states: the **cell state** (long-term memory) and **hidden state** (short-term working memory).
""")

    gates = [
        ("🚪 Forget Gate", "fₜ = σ(Wf·[hₜ₋₁, xₜ] + bf)",
         "Decides what to **erase** from cell state. Output 0 = forget everything, 1 = keep everything.",
         "#FF6584"),
        ("📥 Input Gate", "iₜ = σ(Wᵢ·[hₜ₋₁, xₜ] + bᵢ)  ·  tanh(Wc·[hₜ₋₁, xₜ] + bc)",
         "Decides what **new information** to write to cell state. Two parts: what to write (iₜ) and candidate values (c̃ₜ).",
         "#6C63FF"),
        ("🔄 Cell State Update", "Cₜ = fₜ ⊙ Cₜ₋₁ + iₜ ⊙ c̃ₜ",
         "**Forget** some old info + **add** new info. The cell state is the 'highway' for gradients!",
         "#43D9AD"),
        ("📤 Output Gate", "oₜ = σ(Wo·[hₜ₋₁, xₜ] + bo)   hₜ = oₜ ⊙ tanh(Cₜ)",
         "Decides what to **output** as hidden state based on the current cell state.",
         "#FFB347"),
    ]

    for gate_name, gate_formula, gate_desc, color in gates:
        st.markdown(
            f'<div style="background:{color}12;border:1px solid {color}44;border-radius:10px;'
            f'padding:16px 20px;margin:8px 0">'
            f'<div style="color:{color};font-size:16px;font-weight:700;margin-bottom:8px">{gate_name}</div>'
            f'<div style="background:#0d1117;border-radius:6px;padding:10px;margin-bottom:8px;'
            f'font-family:monospace;color:#43D9AD;font-size:13px">{gate_formula}</div>'
            f'<div style="color:#8890B5;font-size:14px">{gate_desc}</div></div>',
            unsafe_allow_html=True,
        )

    tip("The cell state Cₜ is the key: gradients flow through it mostly unimpeded (only element-wise multiplication with the forget gate), solving vanishing gradients!")

    st.markdown("---")
    st.markdown("#### Why LSTM Works: Constant Error Carousel")
    st.markdown("""
The cell state acts as a **'highway'** for gradients:

- **No matrix multiplication** in the cell state path (only element-wise ops)
- Forget gate can be set to 1 → gradient flows completely
- Network **learns** what to remember and what to forget!

This allows LSTMs to learn dependencies across **hundreds of timesteps**.
    """)


# ══════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("### GRU — Gated Recurrent Unit")
    st.markdown("""
The **GRU** (2014) is a simplified version of LSTM with only 2 gates and no separate cell state.
Often achieves similar performance with fewer parameters and faster training.
""")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### GRU Equations")
        formula("zₜ = σ(Wz·[hₜ₋₁, xₜ])   (Update Gate)")
        formula("rₜ = σ(Wr·[hₜ₋₁, xₜ])   (Reset Gate)")
        formula("h̃ₜ = tanh(W·[rₜ⊙hₜ₋₁, xₜ])  (Candidate)")
        formula("hₜ = (1-zₜ)⊙hₜ₋₁ + zₜ⊙h̃ₜ")

    with col2:
        st.markdown("#### GRU vs LSTM")
        comparison = {
            "Feature": ["Gates", "States", "Parameters", "Speed", "Performance", "Best for"],
            "LSTM": ["3 (forget, input, output)", "Cell + Hidden", "More", "Slower", "Better on long sequences", "Long dependencies"],
            "GRU": ["2 (update, reset)", "Hidden only", "Fewer", "Faster", "Similar on short-medium", "Limited data/compute"],
        }
        st.dataframe(comparison, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 🔀 GRU Intuition")
    st.markdown("""
- **Update Gate (z)**: How much of the previous hidden state to keep vs. overwrite. Like forget + input gate combined.
- **Reset Gate (r)**: How much of the previous hidden state to use when computing the new candidate hidden state.

When `zₜ = 0`: completely ignore previous state → fast adaptation  
When `zₜ = 1`: completely copy previous state → perfect memory
    """)


# ══════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown("### 🧪 Sequence Prediction Demo")
    st.markdown("Simulate an RNN-like process: predict the next value in a time series.")

    col1, col2 = st.columns([1, 2])
    with col1:
        seq_type = st.selectbox("Sequence type:", ["Sine Wave", "Square Wave", "Noisy Sine", "Random Walk"])
        seq_len = st.slider("Sequence length", 30, 150, 80)
        noise_level = st.slider("Noise level", 0.0, 1.0, 0.1, 0.05)
        lookback = st.slider("Lookback window", 2, 20, 5)

        st.markdown("""
**How an RNN would work here:**
1. See last `lookback` values
2. Maintain hidden state across them
3. Predict the next value
4. Update weights via BPTT
        """)

    with col2:
        t = np.linspace(0, 4*np.pi, seq_len)
        sequences = {
            "Sine Wave": np.sin(t),
            "Square Wave": np.sign(np.sin(t)),
            "Noisy Sine": np.sin(t) + noise_level * np.random.randn(seq_len),
            "Random Walk": np.cumsum(np.random.randn(seq_len) * 0.1),
        }
        seq = sequences[seq_type]

        # Simple linear prediction for demo
        predictions = []
        for i in range(lookback, len(seq)):
            window = seq[i-lookback:i]
            w = np.exp(-np.arange(lookback, 0, -1) * 0.3)
            w /= w.sum()
            predictions.append(np.dot(w, window) + noise_level * np.random.randn() * 0.05)

        fig, ax = plt.subplots(figsize=(9, 4))
        fig.patch.set_facecolor("#1E2130"); ax.set_facecolor("#1E2130")
        ax.plot(range(seq_len), seq, color="#6C63FF", linewidth=2, label="True sequence")
        ax.plot(range(lookback, seq_len), predictions, color="#FF6584", linewidth=1.5,
                linestyle="--", label=f"Prediction (lookback={lookback})")
        ax.axvline(seq_len * 0.7, color="#FFB347", alpha=0.5, linestyle=":", label="Train/Val split")
        ax.set_title(f"{seq_type} — Sequence Prediction", color="#E8EAF0")
        ax.set_xlabel("Timestep", color="#E8EAF0"); ax.set_ylabel("Value", color="#E8EAF0")
        ax.tick_params(colors="#E8EAF0"); ax.grid(True, color="#3A3F5C", alpha=0.4)
        ax.legend(facecolor="#1E2130", labelcolor="#E8EAF0")
        for sp in ax.spines.values(): sp.set_edgecolor("#3A3F5C")
        st.pyplot(fig)

    mse = np.mean((np.array(predictions) - seq[lookback:])**2)
    st.metric("Prediction MSE", f"{mse:.4f}")


# ══════════════════════════════════════════════════════════
with tabs[5]:
    render_quiz("rnn_quiz", [
        {
            "question": "What is the key advantage of RNNs over standard MLPs for sequence data?",
            "options": [
                "RNNs are always faster to train",
                "RNNs maintain a hidden state that carries information across timesteps",
                "RNNs use less memory",
                "RNNs don't require backpropagation"
            ],
            "answer": 1,
            "explanation": "The hidden state hₜ = f(hₜ₋₁, xₜ) allows RNNs to model dependencies across time — unlike MLPs that treat each input independently."
        },
        {
            "question": "The vanishing gradient problem in RNNs means:",
            "options": [
                "The network forgets to update weights entirely",
                "Gradients become tiny over long sequences, making early timesteps impossible to learn from",
                "The learning rate becomes too large",
                "The network runs out of memory"
            ],
            "answer": 1,
            "explanation": "Multiplying gradients at each timestep: if |Wₕ| < 1, gradients → 0 exponentially. Early inputs have near-zero gradient after many steps."
        },
        {
            "question": "In an LSTM, the 'forget gate' decides:",
            "options": [
                "What new information to add to the cell state",
                "How to compute the output hidden state",
                "What portion of the old cell state to erase",
                "The learning rate for that timestep"
            ],
            "answer": 2,
            "explanation": "fₜ = σ(...) → output in (0,1). Cell update: Cₜ = fₜ ⊙ Cₜ₋₁ + iₜ ⊙ c̃ₜ. fₜ=0 means forget everything; fₜ=1 means keep everything."
        },
        {
            "question": "How does the GRU differ from the LSTM?",
            "options": [
                "GRU has more parameters and better performance",
                "GRU has 2 gates and no separate cell state, making it simpler and often faster",
                "GRU cannot handle long sequences",
                "GRU uses different loss functions"
            ],
            "answer": 1,
            "explanation": "GRU merges cell+hidden state and uses only update+reset gates. Similar performance to LSTM, fewer parameters, faster training."
        },
        {
            "question": "Which RNN architecture type is used for machine translation (English → French)?",
            "options": [
                "One-to-one",
                "One-to-many",
                "Many-to-one",
                "Many-to-many (asynchronous encoder-decoder)"
            ],
            "answer": 3,
            "explanation": "Translation: variable-length input sequence → variable-length output sequence. Uses encoder RNN + decoder RNN (seq2seq architecture)."
        },
    ])
