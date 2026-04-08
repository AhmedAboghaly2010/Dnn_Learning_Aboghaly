import streamlit as st


def init_progress():
    if "progress" not in st.session_state:
        st.session_state.progress = {}
    if "quiz_scores" not in st.session_state:
        st.session_state.quiz_scores = {}
    if "answered" not in st.session_state:
        st.session_state.answered = {}


def render_quiz(quiz_id: str, questions: list):
    """
    questions: list of dicts with keys:
      - question (str)
      - options (list of str)
      - answer (int, 0-based index)
      - explanation (str)
    """
    init_progress()
    key = f"quiz_{quiz_id}"
    if key not in st.session_state:
        st.session_state[key] = {}

    st.markdown("---")
    st.markdown("### 🧪 Knowledge Check")

    correct_count = 0
    total = len(questions)

    for i, q in enumerate(questions):
        qkey = f"{key}_q{i}"
        st.markdown(f"**Q{i+1}: {q['question']}**")

        disabled = qkey in st.session_state[key]

        choice = st.radio(
            f"Select answer for Q{i+1}",
            options=q["options"],
            key=f"{qkey}_radio",
            label_visibility="collapsed",
            disabled=disabled,
        )

        col1, col2 = st.columns([1, 4])
        with col1:
            if not disabled:
                if st.button("Submit", key=f"{qkey}_btn"):
                    selected_idx = q["options"].index(choice)
                    st.session_state[key][qkey] = selected_idx
                    st.rerun()

        if disabled:
            selected_idx = st.session_state[key][qkey]
            if selected_idx == q["answer"]:
                st.success(f"✅ Correct! {q['explanation']}")
                correct_count += 1
            else:
                st.error(
                    f"❌ Incorrect. The right answer is: **{q['options'][q['answer']]}**\n\n{q['explanation']}"
                )
        st.markdown("")

    # Summary
    answered = len(st.session_state[key])
    if answered == total:
        score = sum(
            1
            for i, q in enumerate(questions)
            if st.session_state[key].get(f"{key}_q{i}") == q["answer"]
        )
        pct = int(score / total * 100)
        st.session_state.quiz_scores[quiz_id] = pct

        if pct == 100:
            st.balloons()
            st.success(f"🏆 Perfect score! {score}/{total} — You nailed it!")
        elif pct >= 70:
            st.info(f"👍 Good job! {score}/{total} ({pct}%) — Keep it up!")
        else:
            st.warning(f"📚 {score}/{total} ({pct}%) — Review the material and try again!")

        # Mark module complete
        st.session_state.progress[quiz_id] = pct
