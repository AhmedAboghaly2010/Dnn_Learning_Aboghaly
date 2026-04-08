import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from utils.styles import apply_styles, formula, tip, section
from utils.visualizations import plot_gradient_descent, plot_loss_landscape, plot_training_curves
from utils.quiz_engine import render_quiz, init_progress

st.set_page_config(page_title="Training & Optimization", page_icon="🏋️", layout="wide")
apply_styles()
init_progress()

section("🏋️ Training & Optimization",
        "Backpropagation, loss functions, and optimizers — how networks actually learn", "beginner")

tabs = st.tabs(["📉 Loss Functions", "🔙 Backpropagation", "🚀 Optimizers",
                "⚖️ Regularization", "🧪 Simulate Training", "✅ Quiz"])


# ══════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown("### Loss Functions — Measuring How Wrong We Are")
    st.markdown("The loss (or cost) function measures the gap between predictions and truth. Training = minimizing loss.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 1. Mean Squared Error (MSE)")
        formula("MSE = (1/n) Σᵢ (yᵢ - ŷᵢ)²")
        st.markdown("""
- Used for **regression** problems
- Penalizes large errors quadratically
- Sensitive to outliers
- Output range: [0, ∞)

**Example:** Predicting house prices
        """)

    with col2:
        st.markdown("#### 2. Binary Cross-Entropy")
        formula("L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]")
        st.markdown("""
- Used for **binary classification**
- Heavily penalizes confident wrong answers
- Works with sigmoid output
- Output range: [0, ∞)

**Example:** Spam detection (yes/no)
        """)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("#### 3. Categorical Cross-Entropy")
        formula("L = -Σₖ yₖ · log(p̂ₖ)")
        st.markdown("""
- Used for **multi-class classification**
- y is one-hot encoded
- Works with softmax output
- Standard for image classification

**Example:** Classify digit as 0-9
        """)

    with col4:
        st.markdown("#### 4. Huber Loss")
        formula("L = 0.5(y-ŷ)² if |y-ŷ|<δ, else δ·|y-ŷ|-0.5δ²")
        st.markdown("""
- Hybrid of MSE + MAE
- **Robust to outliers**
- Behaves like MSE near 0, like MAE for large errors
- Good for regression with noisy data
        """)

    st.markdown("---")
    st.markdown("#### 🎨 Loss Comparison: MSE vs Cross-Entropy")
    y_true = 1.0
    p = np.linspace(0.001, 0.999, 200)
    mse_loss = (y_true - p)**2
    ce_loss = -np.log(p)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.patch.set_facecolor("#1E2130")
    for ax, loss, label, color in zip(axes,
                                       [mse_loss, ce_loss],
                                       ["MSE Loss (y=1)", "Cross-Entropy Loss (y=1)"],
                                       ["#6C63FF", "#FF6584"]):
        ax.set_facecolor("#1E2130")
        ax.plot(p, loss, color=color, linewidth=2.5)
        ax.fill_between(p, loss, alpha=0.12, color=color)
        ax.set_title(label, color="#E8EAF0")
        ax.set_xlabel("Predicted p̂", color="#E8EAF0")
        ax.set_ylabel("Loss", color="#E8EAF0")
        ax.tick_params(colors="#E8EAF0")
        ax.grid(True, color="#3A3F5C", alpha=0.4)
        for sp in ax.spines.values(): sp.set_edgecolor("#3A3F5C")
    plt.tight_layout()
    st.pyplot(fig)

    tip("Cross-entropy has steeper gradients than MSE near prediction=0, making training faster for classification!")


# ══════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("### Backpropagation — How Networks Learn")
    st.markdown("""
**Backpropagation** uses the chain rule to compute gradients of the loss with respect to every parameter,
then updates weights to reduce the loss. It goes **backward** through the network.
""")

    st.markdown("#### 🎯 The Big Picture")
    steps = [
        ("1️⃣ Forward Pass", "Compute predictions ŷ by feeding data through the network"),
        ("2️⃣ Compute Loss", "Compare ŷ to true labels y using the loss function"),
        ("3️⃣ Backward Pass", "Apply chain rule to compute ∂L/∂W for every weight"),
        ("4️⃣ Update Weights", "Nudge weights in the direction that reduces loss"),
        ("5️⃣ Repeat", "Iterate over many batches and epochs until convergence"),
    ]

    for step, desc in steps:
        st.markdown(
            f'<div class="concept-box" style="margin:8px 0;padding:12px 18px">'
            f'<span style="font-weight:700;color:#6C63FF">{step}</span>'
            f'<span style="color:#E8EAF0;margin-left:12px">{desc}</span></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("#### 🔢 The Math of Backprop")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Chain Rule Applied:**")
        formula("∂L/∂W¹ = ∂L/∂ŷ · ∂ŷ/∂h² · ∂h²/∂h¹ · ∂h¹/∂W¹")
        st.markdown("""
Each term is a Jacobian or gradient of one layer's output w.r.t. its input.
We multiply these going backward from the loss.
        """)
    with col2:
        st.markdown("**Weight Update Rule:**")
        formula("W ← W - η · ∂L/∂W")
        st.markdown("""
- **η** (eta) = learning rate (how big each step is)
- **∂L/∂W** = gradient (which direction to move)
- The minus sign: we move **opposite** the gradient (downhill)
        """)

    st.markdown("---")
    st.markdown("#### 🎨 Loss Landscape — What We're Navigating")
    st.plotly_chart(plot_loss_landscape(), use_container_width=True)

    tip("The goal of training is to find the lowest point (global minimum) of this landscape. Backprop computes which direction is downhill.")


# ══════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("### Optimizers — Smarter Ways to Descend")
    st.markdown("Vanilla gradient descent works, but modern optimizers are much faster and more reliable.")

    opt_tab1, opt_tab2, opt_tab3, opt_tab4 = st.tabs(["SGD", "Momentum", "Adam", "Comparison"])

    with opt_tab1:
        st.markdown("#### Stochastic Gradient Descent (SGD)")
        formula("W ← W - η · ∂L/∂W")
        st.markdown("""
**Vanilla SGD**: compute gradient, step in that direction.

- **Batch GD**: use all data → slow but stable
- **Stochastic GD**: use 1 sample → fast but noisy
- **Mini-batch GD**: use N samples → best of both worlds ✅

**Problem:** Can oscillate, gets stuck in saddle points, sensitive to learning rate.
        """)

    with opt_tab2:
        st.markdown("#### SGD with Momentum")
        formula("v ← β·v + (1-β)·∂L/∂W")
        formula("W ← W - η·v")
        st.markdown("""
Momentum adds a "velocity" term that **accumulates past gradients**.

- **β** (momentum coefficient) typically 0.9
- Like a ball rolling downhill: builds up speed in consistent directions
- Dampens oscillations in narrow ravines
- Escapes small bumps (local minima)

**Intuition:** If you keep getting the same gradient signal, accelerate!
        """)

    with opt_tab3:
        st.markdown("#### Adam (Adaptive Moment Estimation)")
        formula("m ← β₁m + (1-β₁)·g       (1st moment / momentum)")
        formula("v ← β₂v + (1-β₂)·g²      (2nd moment / RMSprop)")
        formula("W ← W - η · m̂ / (√v̂ + ε)")
        st.markdown("""
Adam combines **momentum** + **adaptive learning rates** per parameter.

- β₁ = 0.9 (momentum decay)
- β₂ = 0.999 (RMS decay)  
- ε = 1e-8 (numerical stability)

**Why it's popular:**
- Works well with little tuning
- Handles sparse gradients
- Adapts learning rate per-parameter

**Default choice** for most deep learning tasks!
        """)

    with opt_tab4:
        st.markdown("#### Visual Comparison")
        np.random.seed(42)
        epochs = np.arange(1, 51)
        sgd_loss = 2.0 * np.exp(-0.04 * epochs) + 0.4 * np.random.randn(50) * 0.1 + 0.3
        momentum_loss = 2.0 * np.exp(-0.07 * epochs) + 0.2 * np.random.randn(50) * 0.08 + 0.1
        adam_loss = 2.0 * np.exp(-0.12 * epochs) + 0.1 * np.random.randn(50) * 0.05 + 0.05
        sgd_loss = np.clip(sgd_loss, 0, 3)
        momentum_loss = np.clip(momentum_loss, 0, 3)
        adam_loss = np.clip(adam_loss, 0, 3)

        fig, ax = plt.subplots(figsize=(9, 4))
        fig.patch.set_facecolor("#1E2130"); ax.set_facecolor("#1E2130")
        ax.plot(epochs, sgd_loss, color="#FF6584", linewidth=2, label="SGD")
        ax.plot(epochs, momentum_loss, color="#FFB347", linewidth=2, label="SGD + Momentum")
        ax.plot(epochs, adam_loss, color="#43D9AD", linewidth=2, label="Adam")
        ax.set_title("Optimizer Convergence Comparison", color="#E8EAF0")
        ax.set_xlabel("Epoch", color="#E8EAF0"); ax.set_ylabel("Loss", color="#E8EAF0")
        ax.tick_params(colors="#E8EAF0"); ax.grid(True, color="#3A3F5C", alpha=0.4)
        ax.legend(facecolor="#1E2130", labelcolor="#E8EAF0")
        for sp in ax.spines.values(): sp.set_edgecolor("#3A3F5C")
        st.pyplot(fig)


# ══════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("### Regularization — Fighting Overfitting")
    st.markdown("""
**Overfitting**: model memorizes training data, fails on new data.
**Regularization**: techniques to make models generalize better.
""")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### L1 & L2 Regularization")
        formula("L₂: Loss_total = Loss + λΣw²")
        formula("L1: Loss_total = Loss + λΣ|w|")
        st.markdown("""
- **L2 (Weight Decay)**: penalizes large weights → smaller, smoother model
- **L1 (Lasso)**: promotes sparse weights (many become exactly 0)
- **λ** controls regularization strength
        """)

    with col2:
        st.markdown("#### Dropout")
        formula("During training: randomly zero out neurons with probability p")
        st.markdown("""
- Typical p = 0.2–0.5 (drop 20–50% of neurons)
- Forces network to not rely on specific neurons
- Acts like training an **ensemble** of many networks
- At test time: scale weights by (1-p), use all neurons

**Dropout is one of the most effective regularizers!**
        """)

    st.markdown("---")
    st.markdown("#### Bias-Variance Tradeoff")

    x = np.linspace(0, 10, 100)
    np.random.seed(0)
    y_true = np.sin(x) + 0.3

    x_train = np.sort(np.random.choice(x, 15, replace=False))
    y_train = np.sin(x_train) + np.random.randn(15) * 0.3

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.patch.set_facecolor("#1E2130")
    titles = ["Underfitting (High Bias)", "Good Fit", "Overfitting (High Variance)"]
    colors = ["#FFB347", "#43D9AD", "#FF6584"]

    for ax, title, color, deg in zip(axes, titles, colors, [1, 4, 14]):
        ax.set_facecolor("#1E2130")
        ax.scatter(x_train, y_train, color="white", s=30, zorder=5)
        coeffs = np.polyfit(x_train, y_train, deg)
        y_fit = np.polyval(coeffs, x)
        ax.plot(x, y_fit, color=color, linewidth=2)
        ax.plot(x, y_true, color="#8890B5", linewidth=1.5, linestyle="--", label="True function")
        ax.set_title(title, color="#E8EAF0", fontsize=10)
        ax.tick_params(colors="#E8EAF0"); ax.grid(True, color="#3A3F5C", alpha=0.4)
        ax.set_ylim(-2, 3)
        for sp in ax.spines.values(): sp.set_edgecolor("#3A3F5C")
    plt.tight_layout()
    st.pyplot(fig)

    tip("Early stopping, data augmentation, and batch normalization are also powerful regularization tools!")


# ══════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown("### 🧪 Simulate Training: Gradient Descent Explorer")

    col1, col2 = st.columns([1, 2])
    with col1:
        lr = st.select_slider("Learning Rate", options=[0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 1.0], value=0.1)
        steps = st.slider("Training Steps", 5, 60, 25)
        st.markdown("""
**Experiment ideas:**
- Very small LR (0.001) → slow convergence
- Good LR (0.1) → smooth descent
- Large LR (0.5–1.0) → oscillation or divergence!
        """)
        if lr >= 0.5:
            st.warning("⚠️ Learning rate may be too large — watch for oscillation!")
        elif lr <= 0.005:
            st.info("ℹ️ Very small learning rate — convergence will be slow.")

    with col2:
        fig = plot_gradient_descent(lr=lr, steps=steps)
        st.pyplot(fig)

    st.markdown("---")
    st.markdown("#### 📊 Simulate Full Training Curves")

    col1, col2 = st.columns([1, 2])
    with col1:
        n_epochs = st.slider("Epochs", 10, 100, 50)
        overfit = st.slider("Overfitting level", 0, 10, 3,
                             help="Higher = model memorizes training data but fails on validation")
        noise = st.slider("Gradient noise", 0, 10, 3)

    with col2:
        np.random.seed(42)
        epochs = np.arange(1, n_epochs + 1)
        train_loss = 2 * np.exp(-0.05 * epochs) + noise * 0.005 * np.random.randn(n_epochs) + 0.05
        val_loss = train_loss + overfit * 0.005 * epochs * 0.5 / n_epochs + overfit * 0.01 * np.random.randn(n_epochs)
        train_loss = np.clip(train_loss, 0.01, 3)
        val_loss = np.clip(val_loss, 0.01, 3)

        fig = plot_training_curves(train_loss, val_loss)
        st.pyplot(fig)

        if overfit > 5:
            st.warning("📈 Validation loss is diverging from training loss — classic overfitting! Try dropout or L2 regularization.")


# ══════════════════════════════════════════════════════════
with tabs[5]:
    render_quiz("training_quiz", [
        {
            "question": "What does the learning rate control?",
            "options": [
                "The number of neurons in each layer",
                "How large each weight update step is during gradient descent",
                "The size of the training dataset",
                "The depth of the network"
            ],
            "answer": 1,
            "explanation": "W ← W - η·∇L. η (learning rate) scales the gradient: too large = diverge, too small = slow."
        },
        {
            "question": "In backpropagation, gradients flow:",
            "options": [
                "Forward from input to output",
                "Randomly throughout the network",
                "Backward from the loss through each layer to the inputs",
                "Only within the output layer"
            ],
            "answer": 2,
            "explanation": "Backprop applies the chain rule starting from the loss, going backward: output layer → hidden → input layer."
        },
        {
            "question": "What problem does dropout solve?",
            "options": [
                "Slow training speed",
                "Vanishing gradients",
                "Overfitting — forces the network to not rely on any single neuron",
                "Underfitting"
            ],
            "answer": 2,
            "explanation": "Dropout randomly disables neurons during training, forcing robust representations and acting as ensemble training."
        },
        {
            "question": "Adam optimizer is popular because:",
            "options": [
                "It uses a fixed learning rate for all parameters",
                "It adapts learning rates per parameter and combines momentum with RMSprop",
                "It never requires tuning",
                "It uses less memory than SGD"
            ],
            "answer": 1,
            "explanation": "Adam = momentum (1st moment) + adaptive learning rate (2nd moment). Works well across many architectures."
        },
        {
            "question": "Which loss function is most appropriate for a 10-class image classification problem?",
            "options": [
                "Mean Squared Error (MSE)",
                "Binary Cross-Entropy",
                "Categorical Cross-Entropy with Softmax output",
                "Huber Loss"
            ],
            "answer": 2,
            "explanation": "Multi-class classification → Softmax output + Categorical Cross-Entropy. Binary CE is only for 2 classes."
        },
    ])
