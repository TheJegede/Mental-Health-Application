"""
Synthetic student wellbeing dataset generator.

Distributions calibrated to:
  - Healthy Minds Study (Lipson et al.) for prevalence and service utilization
  - ACHA-NCHA 2022 for behavioral feature distributions
  - IPEDS 2022 for enrollment demographics
  - IIE Open Doors 2023 for international student rate
  - NCES 2022 for first-gen rate

Label derivation: demographics → behavioral features (with realistic confounds)
→ latent wellbeing score (behavioral only, no direct demographic input)
→ binary label via bisection-tuned threshold at ~18% prevalence.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path


_GENDER_DIST = {
    "female": 0.56,
    "male": 0.40,
    "non_binary": 0.03,
    "other_not_listed": 0.005,
    "prefer_not_to_say": 0.005,
}

_RACE_DIST = {
    "white": 0.50,
    "hispanic_latino": 0.20,
    "black": 0.13,
    "asian": 0.07,
    "multiracial": 0.05,
    "other": 0.02,
    "native_american": 0.01,
    "pacific_islander": 0.005,
    "prefer_not_to_say": 0.015,
}

_FINANCIAL_AID_DIST = {
    "none": 0.30,
    "loans_only": 0.25,
    "pell_grant": 0.25,
    "both": 0.20,
}

_FINANCIAL_STRESS_RATE_BY_AID = {
    "none": 0.12,
    "loans_only": 0.28,
    "pell_grant": 0.35,
    "both": 0.42,
}

FEATURE_WEIGHTS = {
    "engagement_variance": 0.20,
    "sleep_schedule_drift": 0.20,
    "social_activity_decline": -0.18,
    "academic_trend": -0.22,
    "missed_class_streak": 0.18,
    "financial_stress_flag": 0.12,
    "help_seeking_flag": -0.08,
    "self_report_score_normalized": -0.25,
}


def _sample_categorical(dist: dict, n: int, rng: np.random.Generator) -> np.ndarray:
    keys = list(dist.keys())
    probs = list(dist.values())
    return rng.choice(keys, size=n, p=probs)


def generate(
    n: int = 30_000,
    seed: int = 42,
    target_prevalence: float = 0.18,
    output_path: str | None = None,
) -> pd.DataFrame:
    """
    Generate a synthetic student wellbeing dataset.

    Args:
        n: Number of synthetic students.
        seed: Random state for full reproducibility.
        target_prevalence: Target positive class rate for `support_recommended`.
        output_path: If provided, write CSV to this path.

    Returns:
        DataFrame with all documented columns.
    """
    rng = np.random.default_rng(seed)

    # --- 1. Demographics (IPEDS-calibrated) ---
    gender = _sample_categorical(_GENDER_DIST, n, rng)
    race_ethnicity = _sample_categorical(_RACE_DIST, n, rng)
    first_gen = rng.binomial(1, 0.36, n)
    international_student = rng.binomial(1, 0.056, n)
    financial_aid_status = _sample_categorical(_FINANCIAL_AID_DIST, n, rng)

    # --- 2. Behavioral features (conditioned on demographics for realistic confounds) ---

    # Financial stress: rate conditioned on aid status
    fs_rate = np.array([_FINANCIAL_STRESS_RATE_BY_AID[aid] for aid in financial_aid_status])
    financial_stress_flag = rng.binomial(1, fs_rate)

    # Engagement variance: 0–1, higher = more erratic
    # Financial stress and first-gen bump base variance slightly (realistic confound)
    base_ev = 0.25 + 0.08 * financial_stress_flag + 0.04 * first_gen
    engagement_variance = np.clip(rng.beta(2, 5, n) + base_ev * 0.3, 0.0, 1.0)

    # Sleep schedule drift: 0–1, higher = more disrupted
    base_sd = 0.20 + 0.06 * financial_stress_flag
    sleep_schedule_drift = np.clip(rng.beta(2, 6, n) + base_sd * 0.3, 0.0, 1.0)

    # Social activity decline: −1 to 0 (negative = decline in participation)
    base_sad = -0.15 - 0.05 * financial_stress_flag - 0.03 * first_gen
    social_activity_decline = np.clip(
        rng.normal(base_sad, 0.20, n), -1.0, 0.0
    )

    # Academic trend: −1 to 1 (negative = declining grades)
    base_at = 0.10 - 0.10 * financial_stress_flag - 0.05 * first_gen
    academic_trend = np.clip(rng.normal(base_at, 0.25, n), -1.0, 1.0)

    # Missed class streak: 0–14 consecutive absences
    base_streak_lambda = 1.5 + 1.0 * financial_stress_flag + 0.5 * first_gen
    missed_class_streak = np.clip(rng.poisson(base_streak_lambda, n), 0, 14).astype(int)

    # Help seeking: ~12% base; slightly higher for high behavioral-risk students
    # (rough proxy: students with high engagement variance or long absence streak)
    rough_risk = (engagement_variance > 0.6).astype(float) + (missed_class_streak > 4).astype(float)
    hs_rate = np.clip(0.10 + 0.03 * rough_risk, 0.08, 0.20)
    help_seeking_flag = rng.binomial(1, hs_rate)

    # Self-report score: 0–100 (higher = better wellbeing)
    # Negatively correlated with behavioral risk indicators
    risk_signal = (
        engagement_variance * 0.25
        + sleep_schedule_drift * 0.25
        + (-social_activity_decline) * 0.20
        + (missed_class_streak / 14.0) * 0.15
        + financial_stress_flag * 0.15
    )
    # Map risk_signal (roughly 0–1) to self_report inverted: mean ~65, pushed lower by risk
    self_report_score = np.clip(
        rng.normal(75 - 35 * risk_signal, 12, n), 0.0, 100.0
    )

    # --- 3. Latent wellbeing score (behavioral only, no direct demographic term) ---
    # Higher score → more distress → more likely to be labeled positive
    self_report_normalized = (100.0 - self_report_score) / 100.0  # invert: high distress = high value

    latent = (
        FEATURE_WEIGHTS["engagement_variance"] * engagement_variance
        + FEATURE_WEIGHTS["sleep_schedule_drift"] * sleep_schedule_drift
        + FEATURE_WEIGHTS["social_activity_decline"] * social_activity_decline
        + FEATURE_WEIGHTS["academic_trend"] * academic_trend
        + FEATURE_WEIGHTS["missed_class_streak"] * (missed_class_streak / 14.0)
        + FEATURE_WEIGHTS["financial_stress_flag"] * financial_stress_flag
        + FEATURE_WEIGHTS["help_seeking_flag"] * help_seeking_flag
        + FEATURE_WEIGHTS["self_report_score_normalized"] * self_report_normalized
        + rng.normal(0, 0.05, n)  # Gaussian noise
    )

    # --- 4. Label via bisection to hit target prevalence ---
    lo, hi = float(latent.min()), float(latent.max())
    for _ in range(60):
        mid = (lo + hi) / 2.0
        prev = (latent > mid).mean()
        if abs(prev - target_prevalence) < 0.001:
            break
        if prev > target_prevalence:
            lo = mid
        else:
            hi = mid

    threshold = (lo + hi) / 2.0
    support_recommended = (latent > threshold).astype(int)
    actual_prevalence = support_recommended.mean()

    print(f"[synthetic_data] n={n:,} | seed={seed} | threshold={threshold:.4f}")
    print(f"[synthetic_data] target_prevalence={target_prevalence:.3f} | actual={actual_prevalence:.4f}")
    print(f"[synthetic_data] feature_weights={json.dumps(FEATURE_WEIGHTS, indent=2)}")

    df = pd.DataFrame(
        {
            "gender": gender,
            "race_ethnicity": race_ethnicity,
            "first_gen": first_gen,
            "international_student": international_student,
            "financial_aid_status": financial_aid_status,
            "engagement_variance": engagement_variance.round(4),
            "sleep_schedule_drift": sleep_schedule_drift.round(4),
            "social_activity_decline": social_activity_decline.round(4),
            "academic_trend": academic_trend.round(4),
            "missed_class_streak": missed_class_streak,
            "financial_stress_flag": financial_stress_flag,
            "help_seeking_flag": help_seeking_flag,
            "self_report_score": self_report_score.round(2),
            "support_recommended": support_recommended,
        }
    )

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"[synthetic_data] written to {output_path}")

    return df


def demo_score_from_df(df: pd.DataFrame) -> pd.Series:
    """Reproduce latent score formula for dashboard demo mode. Returns Series of scores 0–100."""
    sr_norm = (100.0 - df["self_report_score"]) / 100.0
    latent = (
        FEATURE_WEIGHTS["engagement_variance"] * df["engagement_variance"]
        + FEATURE_WEIGHTS["sleep_schedule_drift"] * df["sleep_schedule_drift"]
        + FEATURE_WEIGHTS["social_activity_decline"] * df["social_activity_decline"]
        + FEATURE_WEIGHTS["academic_trend"] * df["academic_trend"]
        + FEATURE_WEIGHTS["missed_class_streak"] * (df["missed_class_streak"] / 14.0)
        + FEATURE_WEIGHTS["financial_stress_flag"] * df["financial_stress_flag"]
        + FEATURE_WEIGHTS["help_seeking_flag"] * df["help_seeking_flag"]
        + FEATURE_WEIGHTS["self_report_score_normalized"] * sr_norm
    )
    lo, hi = latent.min(), latent.max()
    return ((latent - lo) / (hi - lo) * 100).clip(0, 100)


if __name__ == "__main__":
    repo_root = Path(__file__).parent.parent
    out = repo_root / "data" / "synthetic" / "student_wellbeing.csv"
    generate(output_path=str(out))
