"""
Campus Wellness Navigator — Counselor Dashboard
Streamlit multi-page application entry point.

Pages (auto-discovered from pages/):
  01_overview.py    Coordinator Overview        (staff only)
  02_behavioral.py  Behavioral Risk             (staff only)
  03_nlp_signals.py NLP Distress Signals        (staff only)
  04_chatbot.py     Student Chatbot             (student-facing)
  05_fairness.py    Model Performance & Fairness (staff only)
"""

import streamlit as st

st.set_page_config(
    page_title="Campus Wellness Navigator",
    page_icon="🌿",
    layout="wide",
)

st.title("Campus Wellness Navigator")
st.caption("Counseling Center Coordinator Dashboard")

st.info(
    "**This system augments counselor judgment — it does not replace it.** "
    "All outputs are support signals, not diagnoses. "
    "Every recommendation requires human review before any action is taken."
)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### Coordinator Overview")
    st.markdown(
        "KPI summary, weekly recommendation trends, and demographic distribution "
        "of active support signals."
    )
    st.page_link("pages/01_overview.py", label="Open Overview →")

with col2:
    st.markdown("### Behavioral Risk")
    st.markdown(
        "Student behavioral risk scores, adjustable threshold, per-student SHAP "
        "feature explanations, and review workflow."
    )
    st.page_link("pages/02_behavioral.py", label="Open Behavioral →")

with col3:
    st.markdown("### NLP Distress Signals")
    st.markdown(
        "Text submissions scored for distress. Crisis-flagged content surfaced "
        "at top. LIME highlighting shows which phrases drove each score."
    )
    st.page_link("pages/03_nlp_signals.py", label="Open NLP Signals →")

col4, col5 = st.columns(2)

with col4:
    st.markdown("### Student Chatbot")
    st.markdown(
        "The only student-facing page. Crisis-aware resource navigator "
        "with all four safety layers active. Crisis resources persistently visible."
    )
    st.page_link("pages/04_chatbot.py", label="Open Chatbot →")

with col5:
    st.markdown("### Model Performance & Fairness")
    st.markdown(
        "Classifier metrics, calibration, NLP recall, chatbot safety audit summary, "
        "and bias audit results across all three modules."
    )
    st.page_link("pages/05_fairness.py", label="Open Fairness →")

st.divider()

with st.expander("About this system"):
    st.markdown(
        """
        **Campus Wellness Navigator** is a proof-of-concept portfolio project.
        It is **not** cleared for institutional deployment.
        Deployment would require IRB approval, a clinical advisory board review, and pilot validation.

        **Three coupled modules:**
        - **Behavioral Risk Classifier** — XGBoost trained on 30k-row synthetic behavioral dataset,
          explained per-prediction with SHAP. Outputs a support recommendation score (0–100), never a diagnosis.
        - **NLP Distress Detector** — three-layer pipeline: crisis keyword override (always first) →
          VADER pre-filter → fine-tuned DistilBERT. Explained with LIME. Same crisis-first constraint.
        - **Crisis-Aware Chatbot** — Groq Llama 3.1 8B + ChromaDB RAG over campus resource corpus.
          Four independent safety layers: crisis lexicon → NLP classifier → LLM safety prompt → output filter.

        **Non-negotiable constraints:**
        - Never diagnoses. Outputs use: *distress signal*, *support recommendation*, *check-in suggested*
        - Crisis keyword override runs before any classifier — always, cannot be disabled
        - Audience split: classifier + NLP are counselor-only; chatbot is student-facing
        - Bias audits mandatory per module via Fairlearn and AIF360
        - Synthetic behavioral data is not a shortcut — real student data is not ethically usable

        **Synthetic data note:** The behavioral classifier trains on synthetic data calibrated to
        Healthy Minds Study and ACHA-NCHA published aggregates. No real student records are used.
        """
    )

st.divider()
st.caption(
    "**Crisis resources always available:** "
    "Call or text **988** · Text HOME to **741741** · "
    "UCLA CAPS: **(310) 825-0768** · Emergency: **911**"
)
