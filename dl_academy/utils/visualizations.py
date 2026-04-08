import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import plotly.graph_objects as go
import plotly.express as px


# ── Color palette ──────────────────────────────────────────
COLORS = {
    "primary": "#6C63FF",
    "secondary": "#FF6584",
    "accent": "#43D9AD",
    "warn": "#FFB347",
    "bg": "#0F1117",
    "card": "#1E2130",
    "text": "#E8EAF0",
}


def styled_fig(figsize=(10, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#1E2130")
    ax.set_facecolor("#1E2130")
    ax.tick_params(colors="#E8EAF0")
    ax.xaxis.label.set_color("#E8EAF0")
    ax.yaxis.label.set_color("#E8EAF0")
    ax.title.set_color("#E8EAF0")
    for spine in ax.spines.values():
        spine.set_edgecolor("#3A3F5C")
    ax.grid(True, color="#3A3F5C", alpha=0.4, linestyle="--")
    return fig, ax


# ── Activation Functions ──────────────────────────────────
def plot_activation(name="relu"):
    x = np.linspace(-4, 4, 300)
    funcs = {
        "relu": (np.maximum(0, x), "ReLU  max(0, x)"),
        "sigmoid": (1 / (1 + np.exp(-x)), "Sigmoid  1/(1+e⁻ˣ)"),
        "tanh": (np.tanh(x), "Tanh"),
        "leaky_relu": (np.where(x >= 0, x, 0.1 * x), "Leaky ReLU"),
        "elu": (np.where(x >= 0, x, np.exp(x) - 1), "ELU"),
        "softmax": (np.exp(x) / np.exp(x).sum(), "Softmax (over vector)"),
    }
    y, label = funcs.get(name, funcs["relu"])
    fig, ax = styled_fig()
    ax.plot(x, y, color=COLORS["primary"], linewidth=2.5, label=label)
    ax.axhline(0, color="#3A3F5C", linewidth=1)
    ax.axvline(0, color="#3A3F5C", linewidth=1)
    ax.set_title(f"Activation: {label}", color=COLORS["text"], fontsize=13)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.legend(facecolor="#1E2130", labelcolor=COLORS["text"])
    return fig


# ── Neural Network Diagram ────────────────────────────────
def draw_neural_net(layer_sizes, title="Neural Network"):
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#1E2130")
    ax.set_facecolor("#1E2130")
    ax.axis("off")

    n_layers = len(layer_sizes)
    max_nodes = max(layer_sizes)
    layer_x = np.linspace(0.1, 0.9, n_layers)
    node_radius = 0.03
    node_colors = [COLORS["primary"], *["#3A3F8F"] * (n_layers - 2), COLORS["secondary"]]
    if n_layers == 1:
        node_colors = [COLORS["primary"]]

    positions = []
    for l_idx, (n, x) in enumerate(zip(layer_sizes, layer_x)):
        ys = np.linspace(0.1, 0.9, n)
        positions.append(list(zip([x] * n, ys)))

    # Draw connections
    if n_layers > 1:
        for l in range(n_layers - 1):
            for (x1, y1) in positions[l]:
                for (x2, y2) in positions[l + 1]:
                    ax.plot([x1, x2], [y1, y2], color="#3A3F5C", alpha=0.4, linewidth=0.7, zorder=1)

    # Draw nodes
    label_names = ["Input"] + [f"Hidden {i}" for i in range(1, n_layers - 1)] + ["Output"]
    if n_layers == 1:
        label_names = ["Input"]
    for l_idx, (layer_pos, color) in enumerate(zip(positions, node_colors)):
        for (x, y) in layer_pos:
            circle = plt.Circle((x, y), node_radius, color=color, zorder=2, linewidth=1.5,
                                 edgecolor="white")
            ax.add_patch(circle)
        mid_y = layer_pos[len(layer_pos) // 2][1]
        ax.text(layer_x[l_idx], 0.03, label_names[l_idx], ha="center", va="bottom",
                color=COLORS["text"], fontsize=9)
        ax.text(layer_x[l_idx], 0.96, f"{layer_sizes[l_idx]}n", ha="center", va="top",
                color=COLORS["accent"], fontsize=9)

    ax.set_title(title, color=COLORS["text"], fontsize=13, pad=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return fig


# ── Loss Landscape ─────────────────────────────────────────
def plot_loss_landscape():
    x = np.linspace(-3, 3, 200)
    y = np.linspace(-3, 3, 200)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + Y**2 + np.sin(3 * X) * 0.5 + np.cos(3 * Y) * 0.5

    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y,
                                     colorscale="Viridis", opacity=0.85)])
    fig.update_layout(
        title="Loss Landscape (3D)",
        scene=dict(
            xaxis_title="Weight 1",
            yaxis_title="Weight 2",
            zaxis_title="Loss",
            bgcolor="#1E2130",
            xaxis=dict(gridcolor="#3A3F5C"),
            yaxis=dict(gridcolor="#3A3F5C"),
            zaxis=dict(gridcolor="#3A3F5C"),
        ),
        paper_bgcolor="#1E2130",
        font=dict(color="#E8EAF0"),
        height=450,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


# ── Training Curves ────────────────────────────────────────
def plot_training_curves(train_loss, val_loss, train_acc=None, val_acc=None):
    epochs = list(range(1, len(train_loss) + 1))
    rows = 2 if train_acc else 1
    fig, axes = plt.subplots(1, rows, figsize=(12 if rows == 2 else 7, 4))
    if rows == 1:
        axes = [axes]
    for ax in axes:
        ax.set_facecolor("#1E2130")
    fig.patch.set_facecolor("#1E2130")

    axes[0].plot(epochs, train_loss, color=COLORS["primary"], label="Train Loss", linewidth=2)
    axes[0].plot(epochs, val_loss, color=COLORS["secondary"], label="Val Loss",
                 linewidth=2, linestyle="--")
    axes[0].set_title("Loss Curves", color=COLORS["text"])
    axes[0].set_xlabel("Epoch", color=COLORS["text"])
    axes[0].set_ylabel("Loss", color=COLORS["text"])
    axes[0].legend(facecolor="#1E2130", labelcolor=COLORS["text"])
    axes[0].tick_params(colors=COLORS["text"])
    axes[0].grid(True, color="#3A3F5C", alpha=0.4)
    for sp in axes[0].spines.values():
        sp.set_edgecolor("#3A3F5C")

    if train_acc:
        axes[1].plot(epochs, train_acc, color=COLORS["accent"], label="Train Acc", linewidth=2)
        axes[1].plot(epochs, val_acc, color=COLORS["warn"], label="Val Acc",
                     linewidth=2, linestyle="--")
        axes[1].set_title("Accuracy Curves", color=COLORS["text"])
        axes[1].set_xlabel("Epoch", color=COLORS["text"])
        axes[1].set_ylabel("Accuracy", color=COLORS["text"])
        axes[1].legend(facecolor="#1E2130", labelcolor=COLORS["text"])
        axes[1].tick_params(colors=COLORS["text"])
        axes[1].grid(True, color="#3A3F5C", alpha=0.4)
        for sp in axes[1].spines.values():
            sp.set_edgecolor("#3A3F5C")

    plt.tight_layout()
    return fig


# ── Gradient Descent Animation ─────────────────────────────
def plot_gradient_descent(lr=0.1, steps=30):
    x_vals = np.linspace(-3, 3, 300)
    y_vals = x_vals**2 + 2 * np.sin(2 * x_vals)

    # Simulate GD
    w = 2.5
    path_x, path_y = [w], [w**2 + 2 * np.sin(2 * w)]
    for _ in range(steps):
        grad = 2 * w + 4 * np.cos(2 * w)
        w = w - lr * grad
        path_x.append(w)
        path_y.append(w**2 + 2 * np.sin(2 * w))

    fig, ax = styled_fig()
    ax.plot(x_vals, y_vals, color=COLORS["primary"], linewidth=2.5, label="f(w) = w² + 2sin(2w)")
    ax.plot(path_x, path_y, "o-", color=COLORS["secondary"], linewidth=1.5,
            markersize=5, label=f"GD path (lr={lr})")
    ax.plot(path_x[0], path_y[0], "^", color=COLORS["warn"], markersize=12, label="Start")
    ax.plot(path_x[-1], path_y[-1], "*", color=COLORS["accent"], markersize=14, label="End")
    ax.set_title(f"Gradient Descent  (lr={lr}, {steps} steps)", color=COLORS["text"])
    ax.set_xlabel("Weight w")
    ax.set_ylabel("Loss")
    ax.legend(facecolor="#1E2130", labelcolor=COLORS["text"])
    return fig


# ── Convolution Demo ───────────────────────────────────────
def plot_convolution_demo():
    image = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ], dtype=float)

    kernel = np.array([[-1, -1, -1],
                       [-1,  8, -1],
                       [-1, -1, -1]])

    # Manual convolution
    h, w = image.shape
    kh, kw = kernel.shape
    out_h, out_w = h - kh + 1, w - kw + 1
    output = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            output[i, j] = np.sum(image[i:i+kh, j:j+kw] * kernel)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.patch.set_facecolor("#1E2130")
    titles = ["Input Image", "Edge Detection Kernel", "Feature Map (Output)"]
    data = [image, kernel, output]
    cmaps = ["Blues", "RdBu", "plasma"]

    for ax, d, t, cmap in zip(axes, data, titles, cmaps):
        ax.set_facecolor("#1E2130")
        im = ax.imshow(d, cmap=cmap, aspect="auto")
        ax.set_title(t, color=COLORS["text"], fontsize=11)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    return fig


# ── Attention Heatmap ──────────────────────────────────────
def plot_attention_heatmap(tokens=None):
    if tokens is None:
        tokens = ["The", "cat", "sat", "on", "the", "mat"]
    n = len(tokens)
    np.random.seed(42)
    attn = np.random.dirichlet(np.ones(n), size=n)

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor("#1E2130")
    ax.set_facecolor("#1E2130")
    im = ax.imshow(attn, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(tokens, color=COLORS["text"], rotation=30)
    ax.set_yticklabels(tokens, color=COLORS["text"])
    ax.set_title("Self-Attention Heatmap", color=COLORS["text"], fontsize=13)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{attn[i,j]:.2f}", ha="center", va="center",
                    color="black" if attn[i, j] > 0.3 else "white", fontsize=8)
    plt.colorbar(im, ax=ax, label="Attention Weight")
    plt.tight_layout()
    return fig
