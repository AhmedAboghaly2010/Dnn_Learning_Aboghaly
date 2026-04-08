import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from utils.styles import apply_styles, formula, tip, section
from utils.visualizations import draw_neural_net, plot_activation
from utils.quiz_engine import render_quiz, init_progress

st.set_page_config(page_title="Neural Networks", page_icon="⚡", layout="wide")
apply_styles()
init_progress()

section("⚡ Neural Networks",
        "Perceptrons, layers, activations — how brains inspire machines", "beginner")

tabs = st.tabs(["🧠 Perceptron", "🏗️ Architecture", "🔥 Activations", "🔬 Build Your Net", "✅ Quiz"])


# ══════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown("### The Perceptron — The Simplest Neuron")
    st.markdown("""
Inspired by biological neurons, a **perceptron** receives inputs, multiplies them by weights,
adds a bias, and passes the result through an activation function.
""")

    col1, col2 = st.columns([1.2, 1])
    with col1:
        # Draw a single neuron
        fig, ax = plt.subplots(figsize=(7, 5))
        fig.patch.set_facecolor("#1E2130"); ax.set_facecolor("#1E2130"); ax.axis("off")

        # Inputs
        inputs = ["x₁", "x₂", "x₃"]
        weights = ["w₁", "w₂", "w₃"]
        iy = [0.8, 0.5, 0.2]
        for i, (label, w, y) in enumerate(zip(inputs, weights, iy)):
            ax.annotate("", xy=(0.45, 0.5), xytext=(0.05, y),
                        arrowprops=dict(arrowstyle="->", color="#8890B5", lw=1.5))
            ax.text(0.03, y, label, color="#6C63FF", fontsize=13, fontweight="bold", va="center")
            ax.text(0.25, (y + 0.5)/2 + 0.02, w, color="#FFB347", fontsize=10, ha="center")

        # Neuron body
        circle = plt.Circle((0.55, 0.5), 0.1, color="#6C63FF", zorder=5)
        ax.add_patch(circle)
        ax.text(0.55, 0.5, "Σ+b", color="white", fontsize=11, fontweight="bold",
                ha="center", va="center", zorder=6)

        # Activation box
        rect = plt.Rectangle((0.7, 0.42), 0.12, 0.16, color="#FF6584", zorder=5)
        ax.add_patch(rect)
        ax.text(0.76, 0.5, "f", color="white", fontsize=13, fontweight="bold",
                ha="center", va="center", zorder=6)

        # Output
        ax.annotate("", xy=(0.95, 0.5), xytext=(0.82, 0.5),
                    arrowprops=dict(arrowstyle="->", color="#43D9AD", lw=2))
        ax.text(0.96, 0.5, "ŷ", color="#43D9AD", fontsize=14, fontweight="bold", va="center")

        ax.text(0.55, 0.38, "bias b", color="#FFB347", fontsize=9, ha="center")
        ax.set_xlim(0, 1.1); ax.set_ylim(0, 1)
        ax.set_title("A Single Neuron (Perceptron)", color="#E8EAF0", fontsize=13)
        st.pyplot(fig)

    with col2:
        st.markdown("#### What Happens Inside?")
        formula("z = w₁x₁ + w₂x₂ + w₃x₃ + b")
        formula("ŷ = f(z)")
        st.markdown("""
**Step by step:**
1. **Multiply** each input by its weight
2. **Sum** them all up + add bias `b`
3. **Apply** activation function `f`

**Weights** = what the neuron has learned  
**Bias** = shifts the decision boundary  
**Activation** = adds non-linearity (crucial!)
        """)
        tip("Without activation functions, stacking layers = just one big linear transformation. Activations make deep networks powerful!")


# ══════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("### Network Architecture")
    st.markdown("""
A **deep neural network** stacks multiple layers. Data flows forward through each layer,
transforming from raw input to a final prediction.
""")

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("#### Layer Types")
        st.markdown("""
| Layer | Role |
|-------|------|
| **Input** | Raw features (pixels, text tokens, sensor readings) |
| **Hidden** | Learned representations — the "thinking" layers |
| **Output** | Final prediction (class probabilities, a number, etc.) |
""")

    with c2:
        st.markdown("#### Common Architectures")
        st.markdown("""
| Name | Layers | Use Case |
|------|--------|----------|
| Shallow Net | 1 hidden | Simple classification |
| Deep MLP | 3-10 hidden | Tabular data, NLP |
| Very Deep | 10-100+ | Images, language |
| ResNet-152 | 152 layers | ImageNet champion |
        """)

    st.markdown("---")
    st.markdown("#### 🎨 Interactive Network Diagram")
    col1, col2 = st.columns([1, 2])
    with col1:
        n_input = st.slider("Input neurons", 1, 8, 4)
        n_h1 = st.slider("Hidden layer 1", 1, 8, 5)
        n_h2 = st.slider("Hidden layer 2", 1, 8, 4)
        n_h3 = st.slider("Hidden layer 3 (0 = skip)", 0, 6, 0)
        n_output = st.slider("Output neurons", 1, 5, 2)

        layers = [n_input, n_h1, n_h2]
        if n_h3 > 0:
            layers.append(n_h3)
        layers.append(n_output)

        total_params = 0
        for i in range(len(layers) - 1):
            total_params += layers[i] * layers[i+1] + layers[i+1]
        st.metric("Total Parameters", f"{total_params:,}")

    with col2:
        fig = draw_neural_net(layers, title=f"Neural Network: {' → '.join(map(str, layers))}")
        st.pyplot(fig)

    st.markdown("---")
    st.markdown("#### Forward Pass — Data Flowing Through the Network")
    formula("h¹ = f(W¹·x + b¹)   →   h² = f(W²·h¹ + b²)   →   ŷ = softmax(W³·h² + b³)")
    st.markdown("""
Each layer receives the **previous layer's output** as its input.
The network progressively builds more **abstract representations**:
- Layer 1: detects simple patterns
- Layer 2: combines patterns into features  
- Layer 3+: high-level concepts
    """)


# ══════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("### Activation Functions")
    st.markdown("""
Activation functions introduce **non-linearity**. Without them, deep networks collapse to linear models.
""")

    act_choice = st.selectbox("Choose an activation function to explore:",
                               ["relu", "sigmoid", "tanh", "leaky_relu", "elu"])

    col1, col2 = st.columns([1.5, 1])
    with col1:
        fig = plot_activation(act_choice)
        st.pyplot(fig)
    with col2:
        descriptions = {
            "relu": {
                "name": "ReLU (Rectified Linear Unit)",
                "formula": "f(x) = max(0, x)",
                "pros": ["Simple & fast to compute", "Avoids vanishing gradient (positive side)", "Sparse activation"],
                "cons": ["'Dying ReLU' problem (neurons stuck at 0)", "Not zero-centered"],
                "use": "Default choice for hidden layers in most architectures"
            },
            "sigmoid": {
                "name": "Sigmoid",
                "formula": "f(x) = 1 / (1 + e⁻ˣ)",
                "pros": ["Outputs between 0-1 (probabilities)", "Smooth gradient"],
                "cons": ["Vanishing gradient for large |x|", "Computationally expensive", "Not zero-centered"],
                "use": "Binary classification output layer"
            },
            "tanh": {
                "name": "Tanh (Hyperbolic Tangent)",
                "formula": "f(x) = (eˣ - e⁻ˣ) / (eˣ + e⁻ˣ)",
                "pros": ["Zero-centered (better than sigmoid)", "Outputs between -1 and 1"],
                "cons": ["Still suffers from vanishing gradient", "Computationally expensive"],
                "use": "RNNs/LSTMs, when zero-centering matters"
            },
            "leaky_relu": {
                "name": "Leaky ReLU",
                "formula": "f(x) = x if x≥0, else 0.01x",
                "pros": ["Fixes dying ReLU problem", "Still simple to compute"],
                "cons": ["Slope for negatives is a hyperparameter"],
                "use": "When dying ReLU is a concern"
            },
            "elu": {
                "name": "ELU (Exponential Linear Unit)",
                "formula": "f(x) = x if x≥0, else α(eˣ-1)",
                "pros": ["Smooth everywhere", "Mean activations closer to zero"],
                "cons": ["More expensive than ReLU", "Has hyperparameter α"],
                "use": "Deep networks where training stability matters"
            },
        }
        d = descriptions[act_choice]
        st.markdown(f"**{d['name']}**")
        formula(d["formula"])
        st.markdown("**✅ Pros:**")
        for p in d["pros"]:
            st.markdown(f"- {p}")
        st.markdown("**⚠️ Cons:**")
        for c in d["cons"]:
            st.markdown(f"- {c}")
        st.markdown(f"**🎯 Use when:** {d['use']}")

    st.markdown("---")
    st.markdown("#### 🎨 All Activations Side by Side")
    x = np.linspace(-4, 4, 300)
    acts = {
        "ReLU": np.maximum(0, x),
        "Sigmoid": 1/(1+np.exp(-x)),
        "Tanh": np.tanh(x),
        "Leaky ReLU": np.where(x >= 0, x, 0.1*x),
    }
    colors = ["#6C63FF", "#FF6584", "#43D9AD", "#FFB347"]
    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor("#1E2130"); ax.set_facecolor("#1E2130")
    for (name, y), color in zip(acts.items(), colors):
        ax.plot(x, y, color=color, linewidth=2, label=name)
    ax.axhline(0, color="#3A3F5C"); ax.axvline(0, color="#3A3F5C")
    ax.set_ylim(-2, 3); ax.grid(True, color="#3A3F5C", alpha=0.4)
    ax.tick_params(colors="#E8EAF0"); ax.set_title("Activation Functions Comparison", color="#E8EAF0")
    ax.legend(facecolor="#1E2130", labelcolor="#E8EAF0", ncol=2)
    for sp in ax.spines.values(): sp.set_edgecolor("#3A3F5C")
    st.pyplot(fig)


# ══════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("### 🔬 Build & Simulate a Network")
    st.markdown("Set custom weights and see the forward pass in action!")

    st.markdown("#### Simple 2→2→1 Network")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Input**")
        x1 = st.slider("x₁", -2.0, 2.0, 0.5, 0.1)
        x2 = st.slider("x₂", -2.0, 2.0, -0.3, 0.1)
        x = np.array([x1, x2])

    with col2:
        st.markdown("**Hidden Layer Weights**")
        w11 = st.slider("w₁₁", -2.0, 2.0, 0.8, 0.1)
        w12 = st.slider("w₁₂", -2.0, 2.0, -0.5, 0.1)
        w21 = st.slider("w₂₁", -2.0, 2.0, 0.3, 0.1)
        w22 = st.slider("w₂₂", -2.0, 2.0, 0.9, 0.1)
        b1, b2 = st.slider("b₁ (hidden)", -1.0, 1.0, 0.1, 0.1), st.slider("b₂ (hidden)", -1.0, 1.0, -0.2, 0.1)

    with col3:
        st.markdown("**Output Layer Weights**")
        wo1 = st.slider("wₒ₁", -2.0, 2.0, 1.0, 0.1)
        wo2 = st.slider("wₒ₂", -2.0, 2.0, -0.7, 0.1)
        bo = st.slider("bₒ (output)", -1.0, 1.0, 0.0, 0.1)

    # Forward pass
    W1 = np.array([[w11, w12], [w21, w22]])
    b_h = np.array([b1, b2])
    W2 = np.array([[wo1, wo2]])
    b_out = np.array([bo])

    z1 = W1 @ x + b_h
    h1 = np.maximum(0, z1)  # ReLU
    z2 = W2 @ h1 + b_out
    y_hat = 1 / (1 + np.exp(-z2[0]))  # Sigmoid output

    st.markdown("---")
    st.markdown("#### 🔍 Forward Pass Results")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("z₁ (pre-activation)", f"[{z1[0]:.2f}, {z1[1]:.2f}]")
    r2.metric("h₁ = ReLU(z₁)", f"[{h1[0]:.2f}, {h1[1]:.2f}]")
    r3.metric("z₂ (output pre-act.)", f"{z2[0]:.3f}")
    r4.metric("ŷ = σ(z₂)", f"{y_hat:.3f}")

    st.markdown(f"""
**Interpretation:** The network predicts **{'class 1' if y_hat > 0.5 else 'class 0'}** 
with **{max(y_hat, 1-y_hat)*100:.1f}% confidence** (threshold = 0.5).
    """)


# ══════════════════════════════════════════════════════════
with tabs[4]:
    render_quiz("nn_quiz", [
        {
            "question": "What is the role of the bias term in a neuron?",
            "options": [
                "It scales the weights",
                "It shifts the activation threshold, allowing the neuron to fire even when all inputs are 0",
                "It prevents overfitting",
                "It normalizes the output"
            ],
            "answer": 1,
            "explanation": "Bias b allows the activation function to shift, making the neuron flexible regardless of input values."
        },
        {
            "question": "Why are activation functions necessary in neural networks?",
            "options": [
                "They speed up training",
                "They reduce memory usage",
                "They add non-linearity, enabling networks to learn complex patterns",
                "They normalize the gradients"
            ],
            "answer": 2,
            "explanation": "Without non-linear activations, stacking layers is equivalent to a single linear transformation."
        },
        {
            "question": "What is the 'dying ReLU' problem?",
            "options": [
                "ReLU is too slow to compute",
                "ReLU neurons can get stuck outputting 0 permanently when weights push inputs below 0",
                "ReLU causes exploding gradients",
                "ReLU cannot model XOR"
            ],
            "answer": 1,
            "explanation": "If inputs to ReLU are consistently negative, gradient = 0 → the neuron never updates → it 'dies'."
        },
        {
            "question": "In the forward pass, data flows:",
            "options": [
                "From output layer to input layer",
                "Randomly between layers",
                "From input layer through hidden layers to output layer",
                "Only through the loss function"
            ],
            "answer": 2,
            "explanation": "Forward pass: input → hidden layers → output. Each layer transforms the representation."
        },
        {
            "question": "Which activation function is best for binary classification outputs?",
            "options": [
                "ReLU (outputs 0 or positive number)",
                "Tanh (outputs -1 to 1)",
                "Sigmoid (outputs 0 to 1, interpretable as probability)",
                "Leaky ReLU"
            ],
            "answer": 2,
            "explanation": "Sigmoid maps any input to (0,1), giving a probability. Ideal for binary classification output neurons."
        },
    ])
