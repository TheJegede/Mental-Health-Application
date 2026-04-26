"""
Behavioral Risk Recommendations — sortable student list, SHAP detail, review workflow.
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
    page_title="Behavioral Risk — Campus Wellness Navigator",
    page_icon="🎯",
    layout="wide",
)

st.warning(
    "**Staff access only.** This page contains confidential student behavioral data. "
    "Not for student access."
)

_SYNTH_CSV = repo_root / "data" / "synthetic" / "student_wellbeing.csv"
_MODEL_PATH = repo_root / "models" / "risk_classifier.pkl"

FEATURE_COLS = [
    "engagement_variance",
    "sleep_schedule_drift",
    "social_activity_decline",
    "academic_trend",
    "missed_class_streak",
    "financial_stress_flag",
    "help_seeking_flag",
    "self_report_score",
]

FEATURE_LABELS = {
    "engagement_variance": "LMS Engagement Variance",
    "sleep_schedule_drift": "Sleep Schedule Drift",
    "social_activity_decline": "Social Activity Decline",
    "academic_trend": "Academic Grade Trend",
    "missed_class_streak": "Missed Class Streak (days)",
    "financial_stress_flag": "Financial Stress Flag",
    "help_seeking_flag": "Help-Seeking Flag",
    "self_report_score": "Self-Report Wellbeing Score",
}

SCORE_BANDS = {
    (86, 100): ("Priority Follow-Up", "🔴"),
    (66, 85): ("Outreach Recommended", "🟠"),
    (41, 65): ("Check-In Suggested", "🟡"),
    (0, 40): ("Baseline", "🟢"),
}


def score_band(score: float) -> tuple[str, str]:
    for (lo, hi), (label, icon) in SCORE_BANDS.items():
        if lo <= score <= hi:
            return label, icon
    return "Baseline", "🟢"


@st.cache_data(show_spinner="Loading student data...")
def load_data() -> pd.DataFrame:
    if _SYNTH_CSV.exists():
        df = pd.read_csv(_SYNTH_CSV)
    else:
        from src.synthetic_data import generate
        df = generate(n=30_000, seed=42)
    df.index.name = "student_idx"
    df["student_id"] = [f"STU-{i:05d}" for i in range(len(df))]
    return df


@st.cache_resource(show_spinner="Loading risk classifier...")
def load_model():
    if not _MODEL_PATH.exists():
        return None
    try:
        from src.risk_classifier import load_model as _load
        return _load(_MODEL_PATH)
    except Exception as e:
        st.warning(f"Model load failed: {e}. Showing demo scores.")
        return None


@st.cache_data(show_spinner="Computing risk scores...")
def compute_scores(df: pd.DataFrame, use_model: bool) -> pd.Series:
    if use_model:
        bundle = load_model()
        if bundle is not None:
            try:
                from src.risk_classifier import predict_score
                results = predict_score(bundle, df[FEATURE_COLS], compute_shap=False)
                return pd.Series([r.score for r in results], index=df.index)
            except Exception:
                pass
    # Demo scoring: reproduce latent formula from synthetic_data.py
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
    return ((latent - lo) / (hi - lo) * 100).clip(0, 100).round(1)


# ---------------------------------------------------------------------------
df = load_data()
bundle = load_model()
model_available = bundle is not None

scores = compute_scores(df, use_model=model_available)

if not model_available:
    st.info(
        "**Demo mode:** Risk classifier not yet trained (run notebooks/03_classifier_training.ipynb). "
        "Scores use the synthetic data generation formula as a proxy."
    )

# ---------------------------------------------------------------------------
st.title("Behavioral Risk Recommendations")
st.caption("Counselor-facing student behavioral signal dashboard. Every score requires human review.")

# Controls
col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 2, 1])
with col_ctrl1:
    threshold = st.slider(
        "Minimum score threshold",
        min_value=0,
        max_value=100,
        value=66,
        step=1,
        help="Start conservative (≥66) to avoid alert fatigue. Lower threshold shows more students.",
    )
with col_ctrl2:
    sort_col = st.selectbox(
        "Sort by",
        ["Score (high→low)", "Score (low→high)", "Missed Class Streak", "Self-Report Score"],
    )
with col_ctrl3:
    show_all = st.checkbox("Show all bands", value=False)

# Build display table
df_scored = df.copy()
df_scored["score"] = scores
df_scored["band"], df_scored["band_icon"] = zip(*df_scored["score"].apply(score_band))

if not show_all:
    df_display = df_scored[df_scored["score"] >= threshold].copy()
else:
    df_display = df_scored.copy()

if sort_col == "Score (high→low)":
    df_display = df_display.sort_values("score", ascending=False)
elif sort_col == "Score (low→high)":
    df_display = df_display.sort_values("score", ascending=True)
elif sort_col == "Missed Class Streak":
    df_display = df_display.sort_values("missed_class_streak", ascending=False)
elif sort_col == "Self-Report Score":
    df_display = df_display.sort_values("self_report_score", ascending=True)

st.caption(
    f"Showing {len(df_display):,} of {len(df):,} students "
    f"(threshold ≥ {threshold})"
)

# ---------------------------------------------------------------------------
# Student list
cols_show = ["student_id", "score", "band", "missed_class_streak", "self_report_score", "financial_stress_flag"]
col_labels = {
    "student_id": "Student ID",
    "score": "Risk Score",
    "band": "Band",
    "missed_class_streak": "Absence Streak",
    "self_report_score": "Self-Report",
    "financial_stress_flag": "Financial Stress",
}

# Show top 100 in table for performance
table_df = df_display[cols_show].head(100).rename(columns=col_labels)
table_df["Risk Score"] = table_df["Risk Score"].round(1)

st.dataframe(table_df, use_container_width=True, height=300, hide_index=True)

if len(df_display) > 100:
    st.caption(f"Showing first 100 of {len(df_display):,} matching students.")

st.divider()

# ---------------------------------------------------------------------------
# Per-student detail
st.subheader("Student Detail")
if len(df_display) == 0:
    st.info("No students match the current threshold. Lower the slider to see more students.")
else:
    selected_id = st.selectbox(
        "Select student",
        df_display["student_id"].head(200).tolist(),
        help="Select a student to view their behavioral signals and SHAP explanation.",
    )

    student_row = df_scored[df_scored["student_id"] == selected_id].iloc[0]
    student_score = student_row["score"]
    band_label, band_icon = score_band(student_score)

    dc1, dc2, dc3 = st.columns(3)
    dc1.metric("Risk Score", f"{student_score:.0f} / 100")
    dc2.metric("Band", f"{band_icon} {band_label}")
    dc3.metric("Recommended Action", band_label)

    st.markdown("**Behavioral Signals**")
    feat_df = pd.DataFrame(
        {
            "Feature": [FEATURE_LABELS[f] for f in FEATURE_COLS],
            "Value": [student_row[f] for f in FEATURE_COLS],
            "Direction": [
                "(+) risk" if f not in ("self_report_score", "help_seeking_flag", "academic_trend")
                else "(-) risk"
                for f in FEATURE_COLS
            ],
        }
    )
    st.dataframe(feat_df, use_container_width=True, hide_index=True)

    # SHAP if model available
    if model_available and bundle is not None:
        if st.button("Run SHAP explanation", key="shap_btn"):
            with st.spinner("Computing SHAP..."):
                try:
                    from src.risk_classifier import predict_score
                    import matplotlib.pyplot as plt
                    results = predict_score(bundle, df_scored[df_scored["student_id"] == selected_id][FEATURE_COLS], compute_shap=True)
                    r = results[0]
                    if r.top_features:
                        st.markdown("**Top Contributing Features (SHAP)**")
                        shap_df = pd.DataFrame([
                            {
                                "Feature": FEATURE_LABELS.get(f["feature"], f["feature"]),
                                "Direction": f["direction"],
                                "SHAP Value": round(f["shap_value"], 4),
                            }
                            for f in r.top_features[:5]
                        ])
                        st.dataframe(shap_df, use_container_width=True, hide_index=True)
                    else:
                        st.info("SHAP values not available for this prediction.")
                except Exception as e:
                    st.error(f"SHAP computation failed: {e}")
    else:
        st.caption("SHAP explanations available after model training (notebooks/03_classifier_training.ipynb).")

    # Mark as reviewed
    st.divider()
    if "reviewed" not in st.session_state:
        st.session_state.reviewed = set()

    if selected_id in st.session_state.reviewed:
        st.success(f"{selected_id} marked as reviewed.")
        if st.button("Unmark reviewed"):
            st.session_state.reviewed.discard(selected_id)
            st.rerun()
    else:
        if st.button("Mark as reviewed", type="primary"):
            st.session_state.reviewed.add(selected_id)
            st.rerun()

    if st.session_state.reviewed:
        with st.expander(f"Reviewed this session ({len(st.session_state.reviewed)})"):
            st.write(sorted(st.session_state.reviewed))

st.divider()
st.caption(
    "**Crisis resources:** Call or text **988** · Text HOME to **741741** · "
    "UCLA CAPS: **(310) 825-0768** · Emergency: **911**"
)
