import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
from utils.styles import apply_styles, card, progress_bar
from utils.quiz_engine import init_progress

st.set_page_config(
    page_title="Deep Learning Academy",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_styles()
init_progress()

# ── Sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 DL Academy")
    st.markdown("---")

    modules = [
        ("🔢", "1. Math Foundations",   "math_quiz",        "pages/1_🔢_Math_Foundations.py"),
        ("⚡", "2. Neural Networks",    "nn_quiz",          "pages/2_⚡_Neural_Networks.py"),
        ("🏋️","3. Training & Optim.", "training_quiz",    "pages/3_🏋️_Training.py"),
        ("👁️","4. CNNs",              "cnn_quiz",         "pages/4_👁️_CNNs.py"),
        ("🔄", "5. RNNs & LSTMs",      "rnn_quiz",         "pages/5_🔄_RNNs_and_LSTMs.py"),
        ("🚀", "6. Transformers",       "transformer_quiz", "pages/6_🚀_Transformers.py"),
        ("🎭", "7. Advanced Topics",    "advanced_quiz",    "pages/7_🎭_Advanced_Topics.py"),
    ]

    scores = st.session_state.get("quiz_scores", {})
    completed = sum(1 for _,_,k,_ in modules if scores.get(k, 0) >= 70)
    total_score = sum(scores.get(k, 0) for _,_,k,_ in modules)
    avg = total_score // len(modules) if modules else 0

    st.markdown(f"**Progress: {completed}/{len(modules)} modules**")
    progress_bar(int(completed / len(modules) * 100))
    st.markdown("")

    for icon, name, key, page in modules:
        score = scores.get(key)
        tag = "✅" if (score or 0) >= 70 else ("⚠️" if score is not None else "○")
        st.page_link(page, label=f"{tag} {name}", icon=icon)

    st.markdown("---")
    st.markdown(
        '<div style="color:#8890B5;font-size:12px">Built with ❤️ using Streamlit</div>',
        unsafe_allow_html=True,
    )


# ── Hero ───────────────────────────────────────────────────
st.markdown(
    """
<div style="text-align:center;padding:40px 20px 30px">
  <div style="font-size:56px;margin-bottom:8px">🧠</div>
  <h1 style="font-size:44px;font-weight:800;color:#E8EAF0;margin:0">
    Deep Learning Academy
  </h1>
  <p style="color:#8890B5;font-size:18px;margin-top:10px;max-width:600px;margin-left:auto;margin-right:auto">
    From absolute zero to advanced practitioner — with interactive visuals,
    hands-on demos, and quizzes at every step.
  </p>
</div>
""",
    unsafe_allow_html=True,
)

# ── Stats row ──────────────────────────────────────────────
scores = st.session_state.get("quiz_scores", {})
completed = sum(1 for _,_,k,_ in modules if scores.get(k, 0) >= 70)
total_score = sum(scores.get(k, 0) for _,_,k,_ in modules)
avg = total_score // len(modules) if scores else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("📚 Modules", f"{len(modules)}")
c2.metric("✅ Completed", f"{completed}/{len(modules)}")
c3.metric("🏆 Avg Score", f"{avg}%")
c4.metric("🔥 Streak", f"{completed} days" if completed else "Start today!")

st.markdown("---")

# ── Module grid ───────────────────────────────────────────
st.markdown("### 📖 Course Modules")
st.markdown("")

module_details = [
    {
        "icon": "🔢",
        "title": "Math Foundations",
        "desc": "Vectors, matrices, dot products, derivatives, and the chain rule — everything you need before touching a neural network.",
        "level": "Beginner",
        "level_color": "#43D9AD",
        "topics": ["Linear Algebra", "Calculus", "Probability"],
        "page": "pages/1_🔢_Math_Foundations.py",
        "key": "math_quiz",
    },
    {
        "icon": "⚡",
        "title": "Neural Networks",
        "desc": "Perceptrons, layers, activation functions, forward pass. Build intuition for how networks represent and transform data.",
        "level": "Beginner",
        "level_color": "#43D9AD",
        "topics": ["Perceptron", "Layers", "Activations"],
        "page": "pages/2_⚡_Neural_Networks.py",
        "key": "nn_quiz",
    },
    {
        "icon": "🏋️",
        "title": "Training & Optimization",
        "desc": "Loss functions, backpropagation, gradient descent, and optimizers. Learn how networks actually learn.",
        "level": "Beginner",
        "level_color": "#43D9AD",
        "topics": ["Backprop", "SGD/Adam", "Regularization"],
        "page": "pages/3_🏋️_Training.py",
        "key": "training_quiz",
    },
    {
        "icon": "👁️",
        "title": "Convolutional Neural Networks",
        "desc": "Filters, feature maps, pooling, and architectures like ResNet. See how CNNs understand images.",
        "level": "Intermediate",
        "level_color": "#FFB347",
        "topics": ["Convolutions", "Pooling", "ResNet"],
        "page": "pages/4_👁️_CNNs.py",
        "key": "cnn_quiz",
    },
    {
        "icon": "🔄",
        "title": "RNNs & LSTMs",
        "desc": "Sequence modeling, vanishing gradients, gates in LSTMs/GRUs. Essential for time-series and NLP.",
        "level": "Intermediate",
        "level_color": "#FFB347",
        "topics": ["RNN", "LSTM", "GRU", "Sequences"],
        "page": "pages/5_🔄_RNNs_and_LSTMs.py",
        "key": "rnn_quiz",
    },
    {
        "icon": "🚀",
        "title": "Transformers & Attention",
        "desc": "Self-attention, multi-head attention, positional encoding. The architecture behind GPT, BERT, and modern AI.",
        "level": "Advanced",
        "level_color": "#FF6584",
        "topics": ["Attention", "BERT", "GPT", "ViT"],
        "page": "pages/6_🚀_Transformers.py",
        "key": "transformer_quiz",
    },
    {
        "icon": "🎭",
        "title": "Advanced Topics",
        "desc": "GANs, VAEs, diffusion models, reinforcement learning, and transfer learning. The frontier of deep learning.",
        "level": "Advanced",
        "level_color": "#FF6584",
        "topics": ["GANs", "VAEs", "Diffusion", "RL"],
        "page": "pages/7_🎭_Advanced_Topics.py",
        "key": "advanced_quiz",
    },
]

cols = st.columns(2)
for i, m in enumerate(module_details):
    col = cols[i % 2]
    score = scores.get(m["key"])
    if score is not None:
        color = "#43D9AD" if score >= 70 else "#FFB347"
        status_html = f'<span style="color:{color};font-weight:700">{score}% score</span>'
    else:
        status_html = '<span style="color:#8890B5">Not started</span>'

    topics_html = " ".join(
        f'<span style="background:#252A3D;color:#8890B5;border-radius:20px;padding:2px 10px;font-size:12px">{t}</span>'
        for t in m["topics"]
    )

    with col:
        st.markdown(
            f"""<div class="module-card {'done' if (score or 0) >= 70 else ''}">
  <div style="display:flex;justify-content:space-between;align-items:flex-start">
    <div>
      <span style="font-size:24px">{m['icon']}</span>
      <span style="font-size:18px;font-weight:700;color:#E8EAF0;margin-left:8px">{m['title']}</span>
    </div>
    <span style="color:{m['level_color']};font-size:12px;font-weight:600;background:rgba(108,99,255,0.08);
    padding:2px 10px;border-radius:20px;border:1px solid {m['level_color']}33">{m['level']}</span>
  </div>
  <p style="color:#8890B5;font-size:14px;margin:10px 0 8px 0">{m['desc']}</p>
  <div style="margin-bottom:8px">{topics_html}</div>
  <div style="font-size:13px;margin-bottom:4px">{status_html}</div>
</div>""",
            unsafe_allow_html=True,
        )
        # ← زرار التنقل الحقيقي تحت كل كارد
        st.page_link(m["page"], label=f"Open {m['title']} →", icon=m["icon"])

st.markdown("---")

# ── How to use ─────────────────────────────────────────────
st.markdown("### 🗺️ How to Use This App")
h1, h2, h3, h4 = st.columns(4)
for col, icon, title, desc in [
    (h1, "📖", "Read", "Study each concept with clear explanations and formulas"),
    (h2, "👁️", "Visualize", "Interact with diagrams and live parameter controls"),
    (h3, "🧪", "Experiment", "Tweak values and see results change in real time"),
    (h4, "✅", "Quiz", "Answer questions to confirm your understanding"),
]:
    with col:
        st.markdown(
            f'<div class="concept-box" style="text-align:center">'
            f'<div style="font-size:32px">{icon}</div>'
            f'<div style="font-weight:700;color:#E8EAF0;margin:8px 0 4px">{title}</div>'
            f'<div style="color:#8890B5;font-size:13px">{desc}</div></div>',
            unsafe_allow_html=True,
        )

st.markdown("")
st.info("👈 **Get started:** Click any module in the left sidebar, or navigate using the pages menu above!")
