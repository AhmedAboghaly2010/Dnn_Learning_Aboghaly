import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from utils.styles import apply_styles, formula, tip, section
from utils.visualizations import plot_convolution_demo
from utils.quiz_engine import render_quiz, init_progress

st.set_page_config(page_title="CNNs", page_icon="👁️", layout="wide")
apply_styles()
init_progress()

section("👁️ Convolutional Neural Networks",
        "How deep learning sees the world — filters, feature maps, and visual hierarchies", "intermediate")

tabs = st.tabs(["🔍 Convolution", "🏊 Pooling", "🏗️ CNN Architecture", "🎨 Feature Maps", "🧪 Interactive", "✅ Quiz"])


# ══════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown("### What is a Convolution?")
    st.markdown("""
A **convolution** slides a small **filter (kernel)** across an image, computing dot products at each position.
This detects patterns **regardless of where they appear** (translation invariance).
""")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### How It Works")
        formula("(I * K)[i,j] = Σₘ Σₙ I[i+m, j+n] · K[m,n]")
        st.markdown("""
1. Place the filter at the top-left of the image
2. Multiply element-wise with the patch underneath
3. Sum all products → one output value (feature map pixel)
4. Slide to next position and repeat

**Parameters:**
- **Kernel size**: e.g., 3×3, 5×5
- **Stride**: how many pixels to move each step
- **Padding**: zeros added around image edges
        """)
    with col2:
        st.markdown("#### Why Not Just Use Dense Layers?")
        st.markdown("""
For a 224×224 image with 3 channels:
- Dense layer: **150,000 input features** → millions of parameters
- Conv layer with 32 filters: only **~900 parameters**!

**CNNs achieve massive parameter savings via:**
- **Local connectivity**: each neuron sees only a small patch
- **Weight sharing**: the same filter slides across the whole image
- **Translation invariance**: detects patterns anywhere in the image
        """)
        tip("A 28×28 MNIST image fully connected to 512 hidden neurons = 401,920 params. A conv layer achieves similar results with ~500!")

    st.markdown("---")
    st.markdown("#### 🎨 Convolution in Action: Edge Detection")
    fig = plot_convolution_demo()
    st.pyplot(fig)
    st.markdown("""
The **edge detection kernel** (Laplacian) highlights boundaries:
```
[[-1, -1, -1],
 [-1,  8, -1],
 [-1, -1, -1]]
```
High response where pixel values change rapidly = **edges**!
    """)


# ══════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("### Pooling — Spatial Downsampling")
    st.markdown("""
After convolution, **pooling** reduces the spatial dimensions while retaining the most important information.
""")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Max Pooling (Most Common)")
        formula("MaxPool[i,j] = max(patch[i:i+k, j:j+k])")
        st.markdown("""
Takes the **maximum** value in each window.

Benefits:
- Reduces spatial size (e.g., 28×28 → 14×14)
- Makes features more position-invariant
- Reduces computation for deeper layers
- Provides some noise robustness
        """)

    with col2:
        st.markdown("#### Average Pooling")
        formula("AvgPool[i,j] = mean(patch[i:i+k, j:j+k])")
        st.markdown("""
Takes the **average** of each window.

- Smoother than max pooling
- Used in: MobileNet, global average pooling at end of network
- **Global Average Pooling** (GAP): reduces entire feature map to a single value — used in modern CNNs instead of Dense layers!
        """)

    st.markdown("---")
    st.markdown("#### 🎨 Max Pooling Visualization")
    # Create a sample feature map
    np.random.seed(7)
    fmap = np.random.randint(0, 10, (8, 8))

    # Apply 2x2 max pooling
    pooled = np.zeros((4, 4), dtype=int)
    for i in range(4):
        for j in range(4):
            pooled[i, j] = fmap[2*i:2*i+2, 2*j:2*j+2].max()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.patch.set_facecolor("#1E2130")
    for ax, data, title in zip(axes, [fmap, pooled], ["Feature Map (8×8)", "After 2×2 Max Pool (4×4)"]):
        ax.set_facecolor("#1E2130")
        im = ax.imshow(data, cmap="viridis", aspect="auto")
        for (r, c), val in np.ndenumerate(data):
            ax.text(c, r, str(val), ha="center", va="center", color="white",
                    fontsize=11 if data.shape[0] == 8 else 14, fontweight="bold")
        ax.set_title(title, color="#E8EAF0", fontsize=11)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    st.pyplot(fig)

    tip("Max pooling cuts spatial dimensions by 2 (with stride=2). This halves width and height → ¼ the data!")


# ══════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("### CNN Architecture Patterns")
    st.markdown("""
Classic CNNs follow a pattern: **alternate convolution+activation with pooling**, 
then flatten and use Dense layers for classification.
""")

    # Architecture diagram
    layers_info = [
        ("Input", "224×224×3", "#4A4A8A"),
        ("Conv2D(64,3×3) + ReLU", "224×224×64", "#6C63FF"),
        ("MaxPool(2×2)", "112×112×64", "#5a52e8"),
        ("Conv2D(128,3×3) + ReLU", "112×112×128", "#6C63FF"),
        ("MaxPool(2×2)", "56×56×128", "#5a52e8"),
        ("Conv2D(256,3×3) + ReLU", "56×56×256", "#6C63FF"),
        ("GlobalAvgPool", "256", "#43D9AD"),
        ("Dense(512) + ReLU", "512", "#FF6584"),
        ("Dense(num_classes) + Softmax", "1000", "#FFB347"),
    ]

    for layer_name, output_shape, color in layers_info:
        st.markdown(
            f'<div style="background:{color}20;border-left:4px solid {color};'
            f'border-radius:0 8px 8px 0;padding:10px 16px;margin:4px 0;display:flex;justify-content:space-between">'
            f'<span style="color:#E8EAF0;font-weight:600">{layer_name}</span>'
            f'<span style="color:{color};font-family:monospace">{output_shape}</span></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("#### Famous CNN Architectures")
    arch_data = {
        "Architecture": ["LeNet-5", "AlexNet", "VGG-16", "ResNet-50", "EfficientNet-B0"],
        "Year": [1998, 2012, 2014, 2015, 2019],
        "Depth": [5, 8, 16, 50, "82 (effective)"],
        "Params": ["60K", "60M", "138M", "25.5M", "5.3M"],
        "Top-1 (ImageNet)": ["N/A", "63.3%", "74.4%", "76.1%", "77.1%"],
        "Key Innovation": [
            "First modern CNN",
            "ReLU, Dropout, GPU",
            "Very deep, simple",
            "Skip connections",
            "Compound scaling"
        ],
    }
    st.dataframe(arch_data, use_container_width=True)

    tip("ResNet introduced **skip connections** that allow gradients to flow directly through layers, enabling very deep networks (50-200+ layers)!")


# ══════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("### What Do CNNs Actually See?")
    st.markdown("""
As data flows through a CNN, each layer builds on the previous to detect increasingly complex features.
""")

    levels = [
        ("Layer 1 (Early)", "🔲", "Edges, corners, simple gradients",
         "Gabor-like filters detect horizontal, vertical, diagonal edges"),
        ("Layer 2 (Early-Mid)", "🔷", "Textures, curves, simple shapes",
         "Combinations of edges form textures like stripes, grids, dots"),
        ("Layer 3 (Mid)", "🔵", "Object parts, complex patterns",
         "Eyes, wheels, leaves — recognizable semantic regions"),
        ("Layer 4 (Deep)", "🟣", "High-level features, semantics",
         "Dog faces, car bodies, specific object concepts"),
        ("Output", "⭕", "Class probabilities",
         "Softmax scores — 'golden retriever: 94%, labrador: 3%, ...'"),
    ]

    for layer, icon, title, desc in levels:
        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown(
                f'<div class="concept-box" style="text-align:center;padding:15px">'
                f'<div style="font-size:28px">{icon}</div>'
                f'<div style="color:#6C63FF;font-size:12px;margin-top:4px">{layer}</div></div>',
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f'<div class="concept-box" style="padding:15px">'
                f'<div style="color:#E8EAF0;font-weight:700;margin-bottom:4px">{title}</div>'
                f'<div style="color:#8890B5;font-size:14px">{desc}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown("#### 🎨 Filter Visualization (Random Filters)")
    st.markdown("Real trained CNN filters look like this (edge detectors, frequency detectors):")

    np.random.seed(123)
    n_filters = 16
    fig, axes = plt.subplots(2, 8, figsize=(14, 4))
    fig.patch.set_facecolor("#1E2130")
    for ax in axes.flat:
        filt = np.random.randn(5, 5)
        ax.imshow(filt, cmap="RdBu_r", aspect="auto")
        ax.axis("off")
    plt.suptitle("Example 5×5 Conv Filters (First Layer)", color="#E8EAF0", fontsize=11)
    plt.tight_layout()
    st.pyplot(fig)


# ══════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown("### 🧪 Interactive Convolution Lab")

    col1, col2 = st.columns([1, 1.5])
    with col1:
        st.markdown("**Choose a kernel:**")
        kernel_type = st.selectbox("Filter Type", [
            "Edge Detection (Laplacian)",
            "Horizontal Edges (Sobel)",
            "Vertical Edges (Sobel)",
            "Blur (Average)",
            "Sharpen",
        ])

        kernels = {
            "Edge Detection (Laplacian)": np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]),
            "Horizontal Edges (Sobel)": np.array([[-1,-2,-1],[0,0,0],[1,2,1]]),
            "Vertical Edges (Sobel)": np.array([[-1,0,1],[-2,0,2],[-1,0,1]]),
            "Blur (Average)": np.ones((3,3)) / 9,
            "Sharpen": np.array([[0,-1,0],[-1,5,-1],[0,-1,0]]),
        }

        k = kernels[kernel_type]
        st.markdown("**Kernel matrix:**")
        fig_k, ax_k = plt.subplots(figsize=(3, 3))
        fig_k.patch.set_facecolor("#1E2130"); ax_k.set_facecolor("#1E2130")
        im = ax_k.imshow(k, cmap="RdBu_r", aspect="auto")
        for (r, c), val in np.ndenumerate(k):
            ax_k.text(c, r, f"{val:.1f}", ha="center", va="center",
                      color="black" if abs(val) < 3 else "white", fontsize=12, fontweight="bold")
        ax_k.axis("off"); ax_k.set_title("Kernel", color="#E8EAF0")
        plt.colorbar(im, ax=ax_k, fraction=0.046)
        st.pyplot(fig_k)

    with col2:
        # Create a test image (checkerboard + circle)
        size = 28
        img = np.zeros((size, size))
        # Draw a circle
        cx, cy, r = size//2, size//2, 8
        for i in range(size):
            for j in range(size):
                if (i-cx)**2 + (j-cy)**2 < r**2:
                    img[i,j] = 1.0

        # Convolve
        kh, kw = k.shape
        oh, ow = size - kh + 1, size - kw + 1
        output = np.zeros((oh, ow))
        for i in range(oh):
            for j in range(ow):
                output[i, j] = np.sum(img[i:i+kh, j:j+kw] * k)

        fig2, axes2 = plt.subplots(1, 2, figsize=(8, 4))
        fig2.patch.set_facecolor("#1E2130")
        for ax, data, title in zip(axes2, [img, output], ["Input", f"After {kernel_type.split('(')[0].strip()}"]):
            ax.set_facecolor("#1E2130")
            ax.imshow(data, cmap="Blues" if data is img else "plasma", aspect="auto")
            ax.set_title(title, color="#E8EAF0", fontsize=10); ax.axis("off")
        plt.tight_layout()
        st.pyplot(fig2)


# ══════════════════════════════════════════════════════════
with tabs[5]:
    render_quiz("cnn_quiz", [
        {
            "question": "What is the main advantage of convolutional layers over fully-connected layers for images?",
            "options": [
                "They are always more accurate",
                "Parameter sharing and local connectivity dramatically reduce parameters",
                "They train faster because they use fewer epochs",
                "They can only handle grayscale images"
            ],
            "answer": 1,
            "explanation": "Weight sharing (same filter slides across image) + local connectivity = orders of magnitude fewer parameters than Dense layers."
        },
        {
            "question": "What does max pooling do?",
            "options": [
                "Applies a filter to detect edges",
                "Increases the spatial resolution",
                "Reduces spatial size by taking the maximum value in each window",
                "Normalizes the feature maps"
            ],
            "answer": 2,
            "explanation": "MaxPool with stride=2 halves both height and width (taking the max in 2×2 windows), reducing computation."
        },
        {
            "question": "What does 'stride' mean in convolution?",
            "options": [
                "The size of the kernel",
                "How many pixels the filter moves at each step",
                "The number of filters applied",
                "The padding added around the image"
            ],
            "answer": 1,
            "explanation": "Stride controls how many pixels the kernel moves at each step. Stride=2 downsamples by 2, stride=1 keeps the same size."
        },
        {
            "question": "What was ResNet's key innovation?",
            "options": [
                "Using very large kernels",
                "Removing all pooling layers",
                "Skip (residual) connections that let gradients flow directly, enabling very deep networks",
                "Training on ImageNet for the first time"
            ],
            "answer": 2,
            "explanation": "ResNet's skip connections y = F(x) + x allow gradients to bypass layers, solving the vanishing gradient in very deep networks."
        },
        {
            "question": "What do early CNN layers typically detect compared to later layers?",
            "options": [
                "Early layers detect complex objects; later layers detect edges",
                "Early layers detect edges/textures; later layers detect high-level semantic features",
                "All layers detect the same features",
                "Only the final layer learns useful features"
            ],
            "answer": 1,
            "explanation": "CNNs learn hierarchically: edges → textures → object parts → semantic concepts as depth increases."
        },
    ])
