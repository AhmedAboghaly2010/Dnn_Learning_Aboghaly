import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from utils.styles import apply_styles, formula, tip, section
from utils.quiz_engine import render_quiz, init_progress

st.set_page_config(page_title="Advanced Topics", page_icon="🎭", layout="wide")
apply_styles()
init_progress()

section("🎭 Advanced Topics",
        "GANs, VAEs, diffusion models, and the frontier of deep learning", "advanced")

tabs = st.tabs(["🎭 GANs", "🔵 VAEs", "🌊 Diffusion", "🔁 Transfer Learning", "🎮 RL Basics", "✅ Quiz"])


# ══════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown("### Generative Adversarial Networks (GANs)")
    st.markdown("""
**GANs** (Goodfellow et al., 2014) train two networks in competition:
a **Generator** that creates fake data and a **Discriminator** that tries to detect fakes.
""")

    col1, col2 = st.columns([1.2, 1])
    with col1:
        # GAN diagram
        fig, ax = plt.subplots(figsize=(9, 5))
        fig.patch.set_facecolor("#1E2130"); ax.set_facecolor("#1E2130"); ax.axis("off")

        # Noise input
        ax.text(0.05, 0.5, "🎲\nNoise z\n~N(0,I)", ha="center", va="center", color="#8890B5",
                fontsize=11, bbox=dict(boxstyle="round", facecolor="#252A3D", edgecolor="#3A3F5C"))
        # Generator
        ax.annotate("", xy=(0.25, 0.5), xytext=(0.14, 0.5),
                    arrowprops=dict(arrowstyle="->", color="#6C63FF", lw=2))
        ax.text(0.33, 0.5, "🧠\nGenerator\nG(z)", ha="center", va="center", color="white",
                fontsize=11, bbox=dict(boxstyle="round", facecolor="#6C63FF", edgecolor="#6C63FF"))
        # Fake image
        ax.annotate("", xy=(0.50, 0.5), xytext=(0.42, 0.5),
                    arrowprops=dict(arrowstyle="->", color="#FFB347", lw=2))
        ax.text(0.54, 0.5, "🖼️\nFake\nImage", ha="center", va="center", color="#FFB347",
                fontsize=10, bbox=dict(boxstyle="round", facecolor="#252A3D", edgecolor="#FFB347"))
        # Real image
        ax.text(0.54, 0.2, "🖼️\nReal\nImage", ha="center", va="center", color="#43D9AD",
                fontsize=10, bbox=dict(boxstyle="round", facecolor="#252A3D", edgecolor="#43D9AD"))
        # Discriminator
        ax.annotate("", xy=(0.68, 0.38), xytext=(0.60, 0.45),
                    arrowprops=dict(arrowstyle="->", color="#8890B5", lw=1.5))
        ax.annotate("", xy=(0.68, 0.35), xytext=(0.60, 0.25),
                    arrowprops=dict(arrowstyle="->", color="#8890B5", lw=1.5))
        ax.text(0.74, 0.37, "🔍\nDiscriminator\nD(x)", ha="center", va="center", color="white",
                fontsize=10, bbox=dict(boxstyle="round", facecolor="#FF6584", edgecolor="#FF6584"))
        # Output
        ax.annotate("", xy=(0.90, 0.37), xytext=(0.81, 0.37),
                    arrowprops=dict(arrowstyle="->", color="#E8EAF0", lw=2))
        ax.text(0.93, 0.37, "Real\nor\nFake?", ha="center", va="center", color="#E8EAF0", fontsize=10)

        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_title("GAN: Generator vs Discriminator", color="#E8EAF0", fontsize=13)
        st.pyplot(fig)

    with col2:
        st.markdown("#### The Minimax Game")
        formula("min_G max_D V(D,G) =")
        formula("𝔼[log D(x)] + 𝔼[log(1-D(G(z)))]")
        st.markdown("""
**Training loop:**
1. **Train Discriminator**: maximize chance of correctly classifying real vs. fake
2. **Train Generator**: minimize chance of Discriminator detecting fakes
3. **Equilibrium**: G generates perfect fakes; D can only guess randomly (50%)

**Problems:**
- **Mode collapse**: G generates only a few types
- **Training instability**: difficult to balance G and D
- **Vanishing gradients**: when D is too good
        """)
        tip("Wasserstein GAN (WGAN) fixes training stability by using a different loss function and a Lipschitz constraint on D!")

    st.markdown("---")
    st.markdown("#### Famous GAN Variants")
    gan_variants = [
        ("DCGAN (2015)", "Deep Convolutional GAN — first stable image GAN"),
        ("CycleGAN (2017)", "Unpaired image-to-image translation (horse ↔ zebra)"),
        ("StyleGAN (2019)", "Photorealistic face generation, style mixing"),
        ("BigGAN (2018)", "High-fidelity class-conditional ImageNet generation"),
        ("Pix2Pix (2017)", "Paired image-to-image (sketch → photo)"),
    ]
    for name, desc in gan_variants:
        st.markdown(
            f'<div style="display:flex;gap:16px;padding:8px 0;border-bottom:1px solid #3A3F5C">'
            f'<span style="color:#6C63FF;font-weight:700;min-width:160px">{name}</span>'
            f'<span style="color:#8890B5">{desc}</span></div>',
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("### Variational Autoencoders (VAEs)")
    st.markdown("""
**VAEs** learn a compressed **latent representation** of data while ensuring the latent space is smooth and continuous — enabling controlled generation.
""")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Autoencoder (standard)")
        st.markdown("""
An **autoencoder** compresses input → **latent code z** → reconstructs output.

```
Input x → Encoder → z → Decoder → x̂
```

Problem: The latent space has **holes** — random z may decode to garbage.
We can't sample new data!
        """)
        formula("Loss = reconstruction_loss(x, x̂)")

    with col2:
        st.markdown("#### VAE: Encoding Distributions")
        st.markdown("""
VAE encodes x into a **distribution** N(μ, σ²) instead of a point.

```
x → Encoder → μ, σ → z ~ N(μ,σ²) → Decoder → x̂
```

The latent space is now **continuous and structured** → we can sample!
        """)
        formula("Loss = reconstruction_loss + KL(q(z|x) || p(z))")
        st.markdown("""
**KL divergence** term: regularizes the latent space to look like N(0,I)
→ ensures the entire space is filled with meaningful samples
        """)

    st.markdown("---")
    st.markdown("#### 🎨 The Reparameterization Trick")
    st.markdown("""
Sampling z ~ N(μ, σ²) is not differentiable → can't backprop through!

**Solution**: z = μ + σ · ε where ε ~ N(0,I)

Now gradients can flow through μ and σ — the randomness is just a constant!
    """)
    formula("z = μ + σ ⊙ ε,  ε ~ N(0, I)")

    st.markdown("---")
    st.markdown("#### 🎨 VAE Latent Space Visualization (2D)")

    # Simulate a 2D latent space
    np.random.seed(42)
    n_classes = 5
    colors = ["#6C63FF", "#FF6584", "#43D9AD", "#FFB347", "#FF8800"]
    class_names = ["cats", "dogs", "cars", "planes", "ships"]

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor("#1E2130"); ax.set_facecolor("#1E2130")

    for i, (color, name) in enumerate(zip(colors, class_names)):
        center = np.array([np.cos(2*np.pi*i/n_classes)*2, np.sin(2*np.pi*i/n_classes)*2])
        points = center + np.random.randn(80, 2) * 0.7
        ax.scatter(points[:,0], points[:,1], color=color, alpha=0.6, s=20, label=name)

    ax.set_title("VAE Latent Space (structured, interpolatable)", color="#E8EAF0")
    ax.tick_params(colors="#E8EAF0"); ax.grid(True, color="#3A3F5C", alpha=0.4)
    ax.legend(facecolor="#1E2130", labelcolor="#E8EAF0")
    for sp in ax.spines.values(): sp.set_edgecolor("#3A3F5C")
    st.pyplot(fig)

    tip("The smooth VAE latent space enables interpolation: moving from one class's region to another generates meaningful in-between samples!")


# ══════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("### Diffusion Models — State of the Art Generation")
    st.markdown("""
**Diffusion models** (DDPM, 2020) are the foundation of Stable Diffusion, DALL-E 2, and Midjourney.
They learn to **reverse a gradual noising process**.
""")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Forward Process (Adding Noise)")
        formula("q(xₜ|xₜ₋₁) = N(xₜ; √(1-βₜ)xₜ₋₁, βₜI)")
        st.markdown("""
Gradually add Gaussian noise over **T timesteps** (e.g. T=1000).
At t=T, the image is pure noise.

This is **fixed** — no learning required here!
        """)

    with col2:
        st.markdown("#### Reverse Process (Denoising)")
        formula("pθ(xₜ₋₁|xₜ) = N(xₜ₋₁; μθ(xₜ,t), Σθ(xₜ,t))")
        st.markdown("""
A neural network **learns to predict the noise** that was added at each step.
Generation = start from pure noise, denoise step by step for T steps.

The network (usually a **U-Net**) is conditioned on the timestep t and optionally on text prompts.
        """)

    st.markdown("---")
    st.markdown("#### 🎨 The Noising/Denoising Process")

    t_steps = 10
    fig, axes = plt.subplots(2, t_steps, figsize=(14, 3))
    fig.patch.set_facecolor("#1E2130")
    np.random.seed(0)

    # Row 1: Forward (add noise)
    img_clean = np.ones((16, 16)) * 0.8
    img_clean[4:12, 4:12] = 0.2
    img_clean[6:10, 6:10] = 0.9

    for i, ax in enumerate(axes[0]):
        ax.set_facecolor("#1E2130")
        noise_level = i / (t_steps - 1)
        noisy = img_clean * (1 - noise_level) + np.random.randn(16, 16) * noise_level
        ax.imshow(np.clip(noisy, 0, 1), cmap="viridis", aspect="auto", vmin=0, vmax=1)
        ax.axis("off")
        if i == 0: ax.set_title("Original", color="#43D9AD", fontsize=8)
        if i == t_steps-1: ax.set_title("Pure Noise", color="#FF6584", fontsize=8)

    # Row 2: Reverse (denoise)
    for i, ax in enumerate(axes[1]):
        ax.set_facecolor("#1E2130")
        noise_level = (t_steps - 1 - i) / (t_steps - 1)
        denoised = img_clean * (1 - noise_level) + np.random.randn(16, 16) * noise_level * 0.5
        ax.imshow(np.clip(denoised, 0, 1), cmap="viridis", aspect="auto", vmin=0, vmax=1)
        ax.axis("off")
        if i == 0: ax.set_title("← Denoise", color="#6C63FF", fontsize=8)
        if i == t_steps-1: ax.set_title("Generated", color="#43D9AD", fontsize=8)

    axes[0][t_steps//2].set_title("Forward →", color="#FFB347", fontsize=9)
    axes[1][t_steps//2].set_title("← Reverse", color="#6C63FF", fontsize=9)
    plt.suptitle("Diffusion: Forward (add noise) and Reverse (denoise)", color="#E8EAF0", fontsize=11)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("#### Key Diffusion Models")
    diffusion_models = [
        ("DDPM (2020)", "Ho et al.", "First practical diffusion model"),
        ("DALL-E 2 (2022)", "OpenAI", "Text-to-image via CLIP + diffusion"),
        ("Stable Diffusion (2022)", "Stability AI", "Open-source latent diffusion"),
        ("Midjourney (2022)", "Midjourney", "Artistic text-to-image"),
        ("Imagen (2022)", "Google", "Photorealistic text-to-image"),
        ("Sora (2024)", "OpenAI", "Video generation with diffusion"),
    ]
    for model, org, desc in diffusion_models:
        st.markdown(
            f'<div style="display:flex;gap:16px;padding:6px 0;border-bottom:1px solid #3A3F5C">'
            f'<span style="color:#6C63FF;font-weight:700;min-width:160px">{model}</span>'
            f'<span style="color:#FFB347;min-width:100px">{org}</span>'
            f'<span style="color:#8890B5">{desc}</span></div>',
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("### Transfer Learning & Fine-Tuning")
    st.markdown("""
**Transfer learning** reuses knowledge from a model trained on one task to solve a different (often related) task.
This is the dominant paradigm in modern deep learning!
""")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Why It Works")
        st.markdown("""
Deep networks learn **hierarchical features**:
- Early layers: universal features (edges, textures)
- Later layers: task-specific features

Early layers trained on ImageNet (millions of images) learn **excellent general visual features** — transferable to X-rays, satellite imagery, etc.!

**Benefits:**
- Less data needed (10x–1000x less!)
- Faster training
- Often better performance than training from scratch
        """)

    with col2:
        st.markdown("#### Strategies")
        strategies = [
            ("Feature Extraction", "Freeze all pretrained layers, only train new head. Fastest, least data needed."),
            ("Fine-Tuning (partial)", "Freeze early layers, unfreeze + retrain later layers. Good middle ground."),
            ("Full Fine-Tuning", "Unfreeze all layers, train with small learning rate. Needs more data."),
            ("LoRA / PEFT", "Add tiny trainable adapter layers — fine-tune LLMs with minimal compute!"),
        ]
        for strategy, desc in strategies:
            st.markdown(
                f'<div style="background:#1E2130;border-left:3px solid #6C63FF;border-radius:0 8px 8px 0;'
                f'padding:10px 14px;margin:6px 0">'
                f'<div style="color:#6C63FF;font-weight:700;margin-bottom:4px">{strategy}</div>'
                f'<div style="color:#8890B5;font-size:13px">{desc}</div></div>',
                unsafe_allow_html=True,
            )

    tip("Rule of thumb: small dataset + similar task → feature extraction. Small dataset + different task → fine-tune fewer layers. Large dataset → fine-tune everything!")


# ══════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown("### Reinforcement Learning Basics")
    st.markdown("""
**RL** trains an **agent** to take actions in an **environment** to maximize cumulative **reward**.
No labeled data — the agent learns from trial and error!
""")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Core Concepts")
        concepts = [
            ("Agent", "The learner/decision-maker (e.g., a neural network)"),
            ("Environment", "The world the agent interacts with (e.g., a game)"),
            ("State (s)", "The agent's current observation of the environment"),
            ("Action (a)", "What the agent does at each step"),
            ("Reward (r)", "Feedback signal: positive for good actions, negative for bad"),
            ("Policy (π)", "The agent's strategy: π(a|s) = probability of action a in state s"),
            ("Value function", "V(s) = expected cumulative reward from state s"),
        ]
        for term, desc in concepts:
            st.markdown(
                f'<div style="display:flex;gap:12px;padding:6px 0;border-bottom:1px solid #3A3F5C">'
                f'<span style="color:#6C63FF;font-weight:700;min-width:120px">{term}</span>'
                f'<span style="color:#8890B5;font-size:13px">{desc}</span></div>',
                unsafe_allow_html=True,
            )

    with col2:
        st.markdown("#### The RL Loop")
        formula("Gₜ = rₜ + γrₜ₊₁ + γ²rₜ₊₂ + ... = Σₖ γᵏrₜ₊ₖ")
        st.markdown("""
**γ** (discount factor): 0 < γ < 1 — future rewards are worth less than immediate rewards.

**Famous RL Algorithms:**
- **Q-Learning / DQN**: learn Q(s,a) value for each state-action pair
- **Policy Gradient (REINFORCE)**: directly optimize the policy
- **PPO (Proximal Policy Optimization)**: stable policy gradient — used in ChatGPT (RLHF)
- **SAC**: soft actor-critic for continuous actions

**RLHF** (RL from Human Feedback) fine-tunes LLMs to follow instructions and be helpful:
GPT → InstructGPT → ChatGPT!
        """)

    st.markdown("---")
    st.markdown("#### 🎮 The Exploration-Exploitation Dilemma")
    st.markdown("""
**Exploration**: try new actions to discover better strategies  
**Exploitation**: use known good actions to maximize current reward

Too much exploration → waste time on bad actions  
Too much exploitation → miss better strategies!

**ε-greedy**: with probability ε, explore randomly; otherwise exploit best known action.
Anneal ε from 1.0 → 0.01 over training.
    """)


# ══════════════════════════════════════════════════════════
with tabs[5]:
    render_quiz("advanced_quiz", [
        {
            "question": "In a GAN, what is the Generator's training objective?",
            "options": [
                "Maximize the probability of correctly classifying real images",
                "Fool the Discriminator into thinking fake images are real",
                "Minimize the reconstruction error directly",
                "Learn the exact distribution of training data explicitly"
            ],
            "answer": 1,
            "explanation": "G learns to generate images D cannot distinguish from real. Loss: G tries to minimize log(1-D(G(z))) or equivalently maximize log(D(G(z)))."
        },
        {
            "question": "What does the KL divergence term in the VAE loss function do?",
            "options": [
                "Measures reconstruction quality",
                "Regularizes the latent space to be close to N(0,I), ensuring a smooth, continuous space",
                "Prevents mode collapse",
                "Acts as a learning rate scheduler"
            ],
            "answer": 1,
            "explanation": "KL(q(z|x) || N(0,I)) forces the encoder distribution to stay near a standard normal — making the latent space smooth for sampling."
        },
        {
            "question": "Diffusion models generate images by:",
            "options": [
                "A generator competing with a discriminator",
                "Encoding to latent space and decoding",
                "Learning to reverse a gradual noising process, denoising from pure noise",
                "Copying and pasting from training data"
            ],
            "answer": 2,
            "explanation": "Diffusion models learn to predict and remove noise at each step. Generation: start from Gaussian noise, denoise T steps → clean image."
        },
        {
            "question": "In transfer learning, 'fine-tuning' means:",
            "options": [
                "Training a new model from random weights",
                "Using the pretrained model with frozen weights as a feature extractor only",
                "Starting from pretrained weights and continuing training on the new task",
                "Distilling a large model into a smaller one"
            ],
            "answer": 2,
            "explanation": "Fine-tuning initializes from pretrained weights and updates them (all or part) on the new task, adapting learned representations."
        },
        {
            "question": "RLHF (Reinforcement Learning from Human Feedback) is used to:",
            "options": [
                "Train image classifiers",
                "Play video games",
                "Fine-tune language models to be helpful and follow instructions based on human preferences",
                "Generate synthetic training data"
            ],
            "answer": 2,
            "explanation": "RLHF (used in InstructGPT/ChatGPT): humans rank model outputs → train reward model → RL (PPO) optimizes LLM to maximize reward."
        },
    ])
