"""
Coordinator Overview — KPI cards, trends, demographic distribution.
Staff access only.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))
load_dotenv(repo_root / ".env")

st.set_page_config(
    page_title="Coordinator Overview — Campus Wellness Navigator",
    page_icon="📊",
    layout="wide",
)

st.warning(
    "**Staff access only.** This page contains confidential student wellbeing data. "
    "Not for student access."
)

_SYNTH_CSV = repo_root / "data" / "synthetic" / "student_wellbeing.csv"

SCORE_BANDS = {
    "priority_follow_up": (86, 100),
    "outreach_recommended": (66, 85),
    "check_in_suggested": (41, 65),
    "baseline": (0, 40),
}


@st.cache_data(show_spinner="Loading student data...")
def load_data() -> pd.DataFrame:
    if _SYNTH_CSV.exists():
        return pd.read_csv(_SYNTH_CSV)
    from src.synthetic_data import generate
    return generate(n=30_000, seed=42)


@st.cache_data(show_spinner=False)
def compute_demo_scores(df: pd.DataFrame) -> pd.Series:
    """Reproduce latent score formula (without trained model) for demo KPIs."""
    sr_norm = (100.0 - df["self_report_score"]) / 100.0
    latent = (
        0.20 * df["engagement_variance"]
        + 0.20 * df["sleep_schedule_drift"]
        + (-0.18) * df["social_activity_decline"]
        + (-0.22) * df["academic_trend"]
        + 0.18 * (df["missed_class_streak"] / 14.0)
        + 0.12 * df["financial_stress_flag"]
        + (-0.08) * df["help_seeking_flag"]
        + (-0.25) * sr_norm
    )
    lo, hi = latent.min(), latent.max()
    return ((latent - lo) / (hi - lo) * 100).clip(0, 100)


df = load_data()
scores = compute_demo_scores(df)

n_total = len(df)
n_check_in = int(((scores >= 41) & (scores < 66)).sum())
n_outreach = int(((scores >= 66) & (scores < 86)).sum())
n_priority = int((scores >= 86).sum())
n_active = n_check_in + n_outreach + n_priority

# Simulated real-time signals (not derivable from static synthetic snapshot)
_rng = np.random.default_rng(99)
n_nlp_7d = 23
n_chatbot_7d = 147
n_crisis_7d = 4

# ---------------------------------------------------------------------------
st.title("Coordinator Overview")
st.caption("Last data refresh: synthetic cohort snapshot · Refreshes weekly in production")

# --- KPI cards ---
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Students", f"{n_total:,}")
c2.metric(
    "Active Recommendations",
    n_active,
    help="Students scoring above 40 on behavioral risk scale",
)
c3.metric(
    "Priority Follow-Up",
    n_priority,
    help="Score 86–100 — highest urgency",
    delta=f"({n_priority / n_total * 100:.1f}% of cohort)",
)
c4.metric("NLP Distress Signals (7d)", n_nlp_7d)
c5.metric(
    "Crisis Routing Events (7d)",
    n_crisis_7d,
    delta="+2 vs prior week",
    delta_color="inverse",
)

st.divider()

# --- Trend chart ---
col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader("Weekly Recommendation Volume")
    st.caption("Simulated 12-week trend — replace with live time-series in production")

    weeks = [f"Wk {i}" for i in range(1, 13)]
    # End-of-semester stress rise weeks 8-12 is realistic
    base_counts = _rng.integers(430, 520, size=12).astype(float)
    base_counts[7:] += _rng.integers(60, 110, size=5)
    trend_df = pd.DataFrame(
        {
            "Check-in Suggested (41–65)": (base_counts * 0.62).astype(int),
            "Outreach Recommended (66–85)": (base_counts * 0.28).astype(int),
            "Priority Follow-Up (86–100)": (base_counts * 0.10).astype(int),
        },
        index=weeks,
    )
    st.line_chart(trend_df, height=280)

with col_right:
    st.subheader("Score Band Distribution")
    band_df = pd.DataFrame(
        {
            "Band": ["Baseline (0–40)", "Check-in (41–65)", "Outreach (66–85)", "Priority (86–100)"],
            "Students": [
                int((scores < 41).sum()),
                n_check_in,
                n_outreach,
                n_priority,
            ],
        }
    ).set_index("Band")
    st.bar_chart(band_df, height=280)

st.divider()

# --- Demographic distribution ---
st.subheader("Recommendation Rate by Demographic Group")
st.caption(
    "Percentage of students in each group with an active support recommendation (score > 40). "
    "Deviations > 5pp warrant fairness review — see Model Performance & Fairness page."
)

df_with_scores = df.copy()
df_with_scores["score"] = scores
df_with_scores["flagged"] = (df_with_scores["score"] > 40).astype(int)

tab1, tab2, tab3, tab4 = st.tabs(["Race/Ethnicity", "Gender", "First-Gen", "Financial Aid"])

with tab1:
    rate = (
        df_with_scores.groupby("race_ethnicity")["flagged"]
        .mean()
        .mul(100)
        .round(1)
        .sort_values(ascending=False)
    )
    overall = df_with_scores["flagged"].mean() * 100
    delta = (rate - overall).round(1)
    disp = pd.DataFrame({"Flag Rate (%)": rate, "Delta from Mean (pp)": delta})
    st.dataframe(disp, use_container_width=True)
    st.bar_chart(rate, height=200)

with tab2:
    rate = (
        df_with_scores.groupby("gender")["flagged"]
        .mean()
        .mul(100)
        .round(1)
        .sort_values(ascending=False)
    )
    st.bar_chart(rate, height=200)

with tab3:
    rate = (
        df_with_scores.groupby("first_gen")["flagged"]
        .mean()
        .mul(100)
        .round(1)
    )
    rate.index = rate.index.map({0: "Continuing-gen", 1: "First-gen"})
    st.bar_chart(rate, height=160)

with tab4:
    rate = (
        df_with_scores.groupby("financial_aid_status")["flagged"]
        .mean()
        .mul(100)
        .round(1)
        .sort_values(ascending=False)
    )
    st.bar_chart(rate, height=180)

st.divider()
st.caption(
    "**Crisis resources:** Call or text **988** · Text HOME to **741741** · "
    "UCLA CAPS: **(310) 825-0768** · Emergency: **911**"
)
