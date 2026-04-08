import streamlit as st

STYLE = """
<style>
/* ── Base ── */
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Hide default Streamlit elements — keep header so sidebar toggle stays visible */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
/* نخلي زرار السايدبار يظهر دايماً */
[data-testid="collapsedControl"] { display: flex !important; }

/* ── Hero cards ── */
.hero-card {
    background: linear-gradient(135deg, #1E2130 0%, #252A3D 100%);
    border: 1px solid #3A3F5C;
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 20px;
}

.module-card {
    background: #1E2130;
    border: 1px solid #3A3F5C;
    border-left: 4px solid #6C63FF;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 14px;
    transition: border-color 0.2s;
}

.module-card.done {
    border-left-color: #43D9AD;
}

.concept-box {
    background: linear-gradient(135deg, #1a1d2e 0%, #1e2238 100%);
    border: 1px solid #3A3F5C;
    border-radius: 12px;
    padding: 20px 24px;
    margin: 16px 0;
}

.formula-box {
    background: #0d1117;
    border: 1px solid #6C63FF;
    border-radius: 8px;
    padding: 16px 20px;
    margin: 12px 0;
    font-family: 'Space Mono', monospace;
    color: #43D9AD;
    font-size: 15px;
    text-align: center;
}

.tip-box {
    background: linear-gradient(90deg, #1a1d2e, #1e2238);
    border-left: 4px solid #FFB347;
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    margin: 12px 0;
    color: #E8EAF0;
}

.success-box {
    background: rgba(67,217,173,0.08);
    border: 1px solid #43D9AD;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 12px 0;
    color: #43D9AD;
}

/* ── Progress bar ── */
.progress-bar-container {
    background: #252A3D;
    border-radius: 20px;
    height: 12px;
    margin: 8px 0 4px 0;
    overflow: hidden;
}
.progress-bar-fill {
    background: linear-gradient(90deg, #6C63FF, #43D9AD);
    height: 100%;
    border-radius: 20px;
    transition: width 0.4s ease;
}

/* ── Section title ── */
.section-title {
    font-size: 28px;
    font-weight: 700;
    color: #E8EAF0;
    margin-bottom: 4px;
}
.section-subtitle {
    color: #8890B5;
    font-size: 15px;
    margin-bottom: 24px;
}

/* ── Badge ── */
.badge {
    display: inline-block;
    background: rgba(108,99,255,0.15);
    color: #6C63FF;
    border: 1px solid #6C63FF;
    border-radius: 20px;
    padding: 2px 12px;
    font-size: 12px;
    font-weight: 600;
    margin-right: 6px;
}
.badge.green { background: rgba(67,217,173,0.12); color: #43D9AD; border-color: #43D9AD; }
.badge.orange { background: rgba(255,179,71,0.12); color: #FFB347; border-color: #FFB347; }
.badge.red { background: rgba(255,101,132,0.12); color: #FF6584; border-color: #FF6584; }

/* ── Streamlit overrides ── */
div[data-testid="stMetricValue"] { color: #6C63FF !important; }
div.stButton > button {
    background: linear-gradient(90deg, #6C63FF, #5a52e8);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 8px 20px;
}
div.stButton > button:hover { background: linear-gradient(90deg, #7B74FF, #6C63FF); }

div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #6C63FF !important;
    border-bottom-color: #6C63FF !important;
}
</style>
"""


def apply_styles():
    st.markdown(STYLE, unsafe_allow_html=True)


def card(content_html, card_class="hero-card"):
    st.markdown(f'<div class="{card_class}">{content_html}</div>', unsafe_allow_html=True)


def formula(text):
    st.markdown(f'<div class="formula-box">{text}</div>', unsafe_allow_html=True)


def tip(text):
    st.markdown(f'<div class="tip-box">💡 {text}</div>', unsafe_allow_html=True)


def success(text):
    st.markdown(f'<div class="success-box">✅ {text}</div>', unsafe_allow_html=True)


def section(title, subtitle="", level="beginner"):
    colors = {"beginner": "#43D9AD", "intermediate": "#FFB347", "advanced": "#FF6584"}
    badge_class = {"beginner": "green", "intermediate": "orange", "advanced": "red"}
    color = colors.get(level, "#6C63FF")
    bc = badge_class.get(level, "")
    st.markdown(
        f"""<div class="section-title">{title}</div>
        <div class="section-subtitle">{subtitle} <span class="badge {bc}">{level.capitalize()}</span></div>""",
        unsafe_allow_html=True,
    )


def progress_bar(pct: int, label=""):
    st.markdown(
        f"""<div style="margin:4px 0">
        <small style="color:#8890B5">{label} {pct}%</small>
        <div class="progress-bar-container">
          <div class="progress-bar-fill" style="width:{pct}%"></div>
        </div></div>""",
        unsafe_allow_html=True,
    )
