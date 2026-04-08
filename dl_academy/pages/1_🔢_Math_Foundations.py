import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from utils.styles import apply_styles, formula, tip, section, card
from utils.quiz_engine import render_quiz, init_progress

st.set_page_config(page_title="Math Foundations", page_icon="🔢", layout="wide")
apply_styles()
init_progress()

section("🔢 Math Foundations",
        "The bedrock of deep learning: vectors, matrices, and calculus", "beginner")

tabs = st.tabs(["📐 Linear Algebra", "📈 Calculus", "🎲 Probability", "🧪 Interactive", "✅ Quiz"])

# ══════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown("### Vectors & Matrices")
    st.markdown("""
Deep learning is fundamentally about **transforming data** using matrices.
A **vector** is a 1D array of numbers. A **matrix** is a 2D array.
""")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
**Vector** — a list of numbers (e.g. features of one data point):
```
x = [height, weight, age]
  = [1.75, 70, 25]
```
Vectors live in **n-dimensional space**. The number of elements = the dimension.
        """)
        formula("x⃗ = [x₁, x₂, …, xₙ]")

    with col2:
        st.markdown("""
**Matrix** — rows of vectors (e.g. a whole dataset):
```
X = [[1.75, 70, 25],
     [1.60, 55, 30],
     [1.80, 85, 22]]
```
Shape: **(3 samples × 3 features)**
        """)
        formula("X ∈ ℝᵐˣⁿ  (m rows, n columns)")

    st.markdown("---")
    st.markdown("### The Dot Product — The Most Important Operation")
    st.markdown("""
The **dot product** is everywhere in deep learning. It computes a weighted sum.
""")
    formula("a⃗ · b⃗ = a₁b₁ + a₂b₂ + … + aₙbₙ = Σᵢ aᵢbᵢ")

    st.markdown("""
**Why does it matter?** A single neuron computes:
`output = dot(weights, inputs) + bias`

This is the foundation of *every* neural network layer!
""")

    # Visual: dot product
    st.markdown("#### 🎨 Visualizing the Dot Product")
    c1, c2 = st.columns(2)
    with c1:
        a = np.array([2.0, 1.0])
        b = np.array([1.0, 2.0])
        dot = np.dot(a, b)
        fig, ax = plt.subplots(figsize=(4.5, 4))
        fig.patch.set_facecolor("#1E2130"); ax.set_facecolor("#1E2130")
        ax.quiver(0, 0, a[0], a[1], angles='xy', scale_units='xy', scale=1,
                  color="#6C63FF", width=0.015, label=f"a = {a}")
        ax.quiver(0, 0, b[0], b[1], angles='xy', scale_units='xy', scale=1,
                  color="#FF6584", width=0.015, label=f"b = {b}")
        ax.set_xlim(-0.5, 3); ax.set_ylim(-0.5, 3)
        ax.axhline(0, color="#3A3F5C"); ax.axvline(0, color="#3A3F5C")
        ax.grid(True, color="#3A3F5C", alpha=0.4)
        ax.tick_params(colors="#E8EAF0")
        ax.set_title(f"a · b = {dot:.0f}", color="#E8EAF0")
        ax.legend(facecolor="#1E2130", labelcolor="#E8EAF0")
        st.pyplot(fig)
    with c2:
        st.markdown(f"""
**a** = [2, 1]  
**b** = [1, 2]

**Dot product:**  
`2×1 + 1×2 = {int(dot)}`

When two vectors point in the **same direction**, the dot product is **large**.  
When they are **perpendicular**, it is **zero**.  
This is how a neuron measures "how much does this input match my weights?"
        """)

    st.markdown("---")
    st.markdown("### Matrix Multiplication")
    formula("C = A × B   where  Cᵢⱼ = Σₖ Aᵢₖ Bₖⱼ")
    st.markdown("""
When a neural network processes a **batch** of inputs, it uses matrix multiplication:
- **X** (batch_size × features) × **W** (features × neurons) = **Z** (batch_size × neurons)

All neurons in a layer are computed **simultaneously** with one matrix multiply — that's why GPUs are so powerful for deep learning!
    """)
    tip("Rows of A dot with columns of B. The inner dimensions must match: (m×k) × (k×n) = (m×n)")


# ══════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("### Calculus for Deep Learning")
    st.markdown("""
Deep learning uses **derivatives** to measure *"how much does the loss change if I tweak a weight slightly?"*
""")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Derivative (1 variable)")
        formula("f'(x) = df/dx = lim[h→0] (f(x+h) - f(x)) / h")
        st.markdown("""
The derivative at a point is the **slope of the tangent line**.

Common rules:
- `d/dx [xⁿ] = n·xⁿ⁻¹`
- `d/dx [sin(x)] = cos(x)`
- `d/dx [eˣ] = eˣ`
        """)

    with col2:
        st.markdown("#### The Chain Rule ⭐")
        formula("d/dx [f(g(x))] = f'(g(x)) · g'(x)")
        st.markdown("""
This is the **backbone of backpropagation**.

If a network has layers `L1 → L2 → L3`, then:

`∂Loss/∂W1 = ∂Loss/∂L3 · ∂L3/∂L2 · ∂L2/∂L1 · ∂L1/∂W1`

We *chain* derivatives backward — hence "backprop"!
        """)

    st.markdown("---")
    st.markdown("#### 🎨 Derivative Visualization")

    x_range = st.slider("x value to examine", -3.0, 3.0, 1.0, 0.1)
    x = np.linspace(-3, 3, 300)
    y = x**3 - 2*x

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#1E2130"); ax.set_facecolor("#1E2130")
    ax.plot(x, y, color="#6C63FF", linewidth=2.5, label="f(x) = x³ - 2x")

    # Tangent line
    dy_at = 3*x_range**2 - 2
    tang_x = np.linspace(x_range - 1, x_range + 1, 50)
    tang_y = (x_range**3 - 2*x_range) + dy_at*(tang_x - x_range)
    ax.plot(tang_x, tang_y, color="#FF6584", linewidth=2, linestyle="--",
            label=f"Tangent (slope={dy_at:.2f})")
    ax.scatter([x_range], [x_range**3 - 2*x_range], color="#43D9AD", s=80, zorder=5)

    ax.axhline(0, color="#3A3F5C"); ax.axvline(0, color="#3A3F5C")
    ax.set_ylim(-6, 6); ax.tick_params(colors="#E8EAF0")
    ax.grid(True, color="#3A3F5C", alpha=0.4)
    ax.set_title(f"f'({x_range:.1f}) = 3x² - 2 = {dy_at:.2f}", color="#E8EAF0")
    ax.legend(facecolor="#1E2130", labelcolor="#E8EAF0")
    for sp in ax.spines.values(): sp.set_edgecolor("#3A3F5C")
    st.pyplot(fig)

    tip("The derivative tells the optimizer which direction to move weights. Negative slope → increase weight; positive slope → decrease weight.")


# ══════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("### Probability & Statistics")
    st.markdown("""
Deep learning models output **probability distributions**. Understanding probability is key to understanding loss functions.
""")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Probability Basics")
        formula("0 ≤ P(X) ≤ 1   and   ΣP(xᵢ) = 1")
        st.markdown("""
**Softmax** turns raw scores into probabilities:
        """)
        formula("softmax(zᵢ) = e^zᵢ / Σⱼ e^zⱼ")
        st.markdown("The output sums to 1 — it's a probability distribution!")

    with col2:
        st.markdown("#### Cross-Entropy Loss")
        formula("L = -Σᵢ yᵢ · log(p̂ᵢ)")
        st.markdown("""
- **yᵢ** = true label (1-hot encoded)
- **p̂ᵢ** = predicted probability

**Intuition**: if the model is confident and **wrong**, loss is huge.  
If the model is confident and **right**, loss is near zero.

This is the standard loss for **classification** tasks.
        """)

    st.markdown("---")
    st.markdown("#### 🎨 Visualizing Cross-Entropy")
    p = np.linspace(0.001, 0.999, 200)
    ce = -np.log(p)
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor("#1E2130"); ax.set_facecolor("#1E2130")
    ax.plot(p, ce, color="#6C63FF", linewidth=2.5)
    ax.fill_between(p, ce, alpha=0.15, color="#6C63FF")
    ax.set_xlabel("Predicted probability p̂ (for true class)", color="#E8EAF0")
    ax.set_ylabel("Loss = -log(p̂)", color="#E8EAF0")
    ax.set_title("Cross-Entropy Loss", color="#E8EAF0")
    ax.tick_params(colors="#E8EAF0")
    ax.grid(True, color="#3A3F5C", alpha=0.4)
    for sp in ax.spines.values(): sp.set_edgecolor("#3A3F5C")
    st.pyplot(fig)

    tip("When probability → 1 (very confident and correct), loss → 0. When probability → 0 (very wrong), loss → ∞!")


# ══════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("### 🧪 Interactive: Matrix Multiplication Explorer")
    st.markdown("Build intuition by multiplying matrices step by step.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Matrix A (2×3)**")
        a_vals = []
        for i in range(2):
            row = st.columns(3)
            a_row = [row[j].number_input(f"A[{i},{j}]", value=float(i*3+j+1),
                                          key=f"a{i}{j}", step=1.0) for j in range(3)]
            a_vals.append(a_row)
        A = np.array(a_vals)

    with c2:
        st.markdown("**Matrix B (3×2)**")
        b_vals = []
        for i in range(3):
            row = st.columns(2)
            b_row = [row[j].number_input(f"B[{i},{j}]", value=float(i*2+j+1),
                                          key=f"b{i}{j}", step=1.0) for j in range(2)]
            b_vals.append(b_row)
        B = np.array(b_vals)

    st.markdown("**Result: C = A × B (2×2)**")
    C = A @ B

    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    fig.patch.set_facecolor("#1E2130")
    titles = [f"A\n{A.shape}", f"B\n{B.shape}", f"C = A×B\n{C.shape}"]
    mats = [A, B, C]
    for ax, mat, title in zip(axes, mats, titles):
        ax.set_facecolor("#1E2130")
        im = ax.imshow(mat, cmap="viridis", aspect="auto")
        for (r, c_), val in np.ndenumerate(mat):
            ax.text(c_, r, f"{val:.0f}", ha="center", va="center",
                    color="white", fontsize=11, fontweight="bold")
        ax.set_title(title, color="#E8EAF0", fontsize=10)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown(f"**Calculation:** A{A.shape} × B{B.shape} = C{C.shape}")


# ══════════════════════════════════════════════════════════
with tabs[4]:
    render_quiz("math_quiz", [
        {
            "question": "What does the dot product of two vectors measure?",
            "options": [
                "Their sum",
                "A weighted sum — how much they align",
                "Their cross product",
                "Their element-wise difference"
            ],
            "answer": 1,
            "explanation": "The dot product computes a⃗·b⃗ = Σ aᵢbᵢ, a weighted sum that measures alignment between vectors."
        },
        {
            "question": "In matrix multiplication A×B, what must be true about their shapes?",
            "options": [
                "Both must be square",
                "A's rows must equal B's columns",
                "A's columns must equal B's rows",
                "Both must have the same shape"
            ],
            "answer": 2,
            "explanation": "For A(m×k) × B(k×n): A's columns (k) must equal B's rows (k). Result is (m×n)."
        },
        {
            "question": "The chain rule in calculus is the mathematical basis for:",
            "options": [
                "Convolutions in CNNs",
                "Forward propagation",
                "Backpropagation",
                "Softmax normalization"
            ],
            "answer": 2,
            "explanation": "Backpropagation applies the chain rule to compute gradients layer by layer, flowing from output back to input."
        },
        {
            "question": "What does cross-entropy loss penalize most?",
            "options": [
                "Correct predictions with high confidence",
                "Predictions far from 0.5",
                "Wrong predictions with high confidence",
                "All predictions equally"
            ],
            "answer": 2,
            "explanation": "-log(p̂) → ∞ as p̂ → 0. If the model is very confident but wrong, the loss explodes."
        },
        {
            "question": "What does the softmax function guarantee?",
            "options": [
                "Outputs are between -1 and 1",
                "Outputs sum to 1 and are all positive (a probability distribution)",
                "The largest input becomes exactly 1",
                "All outputs are equal"
            ],
            "answer": 1,
            "explanation": "softmax(z)ᵢ = eᶻⁱ / Σeᶻʲ. All values > 0 and sum to 1 — a valid probability distribution."
        },
    ])
