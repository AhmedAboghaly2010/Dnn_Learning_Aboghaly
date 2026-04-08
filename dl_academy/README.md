# 🧠 Deep Learning Academy

An interactive, visual deep learning course — from zero to advanced — built with Streamlit.

## 🚀 Live Demo
Deploy to [Streamlit Community Cloud](https://streamlit.io/cloud) in 2 minutes!

## 📚 Course Modules

| Module | Level | Topics |
|--------|-------|--------|
| 🔢 Math Foundations | Beginner | Linear algebra, calculus, probability |
| ⚡ Neural Networks | Beginner | Perceptrons, activations, forward pass |
| 🏋️ Training | Beginner | Backprop, optimizers, regularization |
| 👁️ CNNs | Intermediate | Convolutions, pooling, ResNet |
| 🔄 RNNs & LSTMs | Intermediate | Sequences, gates, vanishing gradient |
| 🚀 Transformers | Advanced | Attention, BERT, GPT |
| 🎭 Advanced | Advanced | GANs, VAEs, Diffusion, RL |

## ✨ Features
- **Visual explanations** with interactive matplotlib/plotly charts
- **Interactive demos** — adjust parameters and see results in real time
- **Quizzes** after every module with instant feedback
- **Progress tracking** across all modules
- **Dark theme** optimized for learning

## 🛠️ Local Setup

```bash
git clone <your-repo>
cd dl_academy
pip install -r requirements.txt
streamlit run app.py
```

## ☁️ Deploy to Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click "New app"
4. Select your repo, branch `main`, and `app.py`
5. Click "Deploy!" 🎉

## 📁 Project Structure

```
dl_academy/
├── app.py                    # Main dashboard
├── requirements.txt
├── .streamlit/
│   └── config.toml           # Dark theme config
├── pages/
│   ├── 1_🔢_Math_Foundations.py
│   ├── 2_⚡_Neural_Networks.py
│   ├── 3_🏋️_Training.py
│   ├── 4_👁️_CNNs.py
│   ├── 5_🔄_RNNs_and_LSTMs.py
│   ├── 6_🚀_Transformers.py
│   └── 7_🎭_Advanced_Topics.py
└── utils/
    ├── styles.py             # CSS & UI helpers
    ├── visualizations.py     # Shared chart functions
    └── quiz_engine.py        # Quiz component
```

## 🤝 Contributing
Pull requests welcome! Add more modules, improve visualizations, or fix bugs.
