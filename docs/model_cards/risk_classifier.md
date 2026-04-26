# Model Card — Behavioral Risk Classifier

**Status:** Complete — Phase 2
**Version:** 1.0.0
**Last updated:** 2026-04-25

---

## Model Details

| Field | Value |
|---|---|
| Model type | XGBoost (primary) / Logistic Regression (baseline) / Random Forest (secondary) |
| Task | Binary classification: `support_recommended` (0/1) |
| Output | `support_recommendation_score` 0–100 (calibrated probability × 100) |
| Training data | `data/synthetic/student_wellbeing.csv` — 30,000 synthetic students, seed=42 |
| Input features | 8 behavioral + survey features (see data dictionary) |
| Explainability | SHAP TreeExplainer — top 3 contributing features per prediction |
| Calibration | Platt scaling (CalibratedClassifierCV, fit on validation set) |
| Class balance | SMOTE on training split only |

---

## Intended Use

**Intended users:** Counseling center coordinators  
**Intended use:** Triage support — surfaces students whose behavioral patterns suggest they may benefit from a wellbeing check-in. One input among many; never used in isolation for action.

**Out-of-scope uses:**
- Clinical diagnosis of any mental health condition
- Automated outreach without human review
- Direct student-facing output
- Any institutional deployment without IRB approval and clinical advisory review

---

## Score Band Interpretation

| Score | Band | Recommended counselor action |
|---|---|---|
| 0–40 | Baseline | No specific action indicated |
| 41–65 | Check-in suggested | Consider proactive outreach when resources allow |
| 66–85 | Outreach recommended | Proactive outreach recommended |
| 86–100 | Priority follow-up | Priority follow-up |

Every score includes: calibrated probability, confidence context, top 3 SHAP-attributed behavioral signals.

---

## Training Approach

- Stratified 70/15/15 split (train/val/test), `random_state=42`
- SMOTE applied to training split only — no leakage into val/test
- GridSearchCV: F1 (positive class) as scoring metric, 5-fold CV
- Platt scaling calibration fit on validation set

---

## Evaluation Metrics

*To be filled after `notebooks/03_classifier_training.ipynb` completes.*

| Metric | Validation | Test |
|---|---|---|
| F1 (positive class) | TBD | TBD |
| ROC-AUC | TBD | TBD |
| Average Precision | TBD | TBD |
| Brier Score | TBD | TBD |
| 5-fold CV F1 | TBD ± TBD | — |

---

## Bias Audit Results

*To be filled after `notebooks/05_classifier_bias_audit.ipynb` completes.*

| Demographic group | Parity spread | TPR spread | FPR spread | Status |
|---|---|---|---|---|
| Gender | TBD | TBD | TBD | TBD |
| Race/ethnicity | TBD | TBD | TBD | TBD |
| First-gen status | TBD | TBD | TBD | TBD |
| International student | TBD | TBD | TBD | TBD |
| Financial aid status | TBD | TBD | TBD | TBD |

**Mitigation applied:** TBD (threshold adjustment / re-weighting / feature exclusion — before/after documented in notebook 05)

**Cultural normality review:** Completed in `notebooks/04_classifier_explainability.ipynb` cell 5. Findings: TBD.

---

## Ethical Constraints

- Output never contains clinical labels or DSM-5 terminology
- SHAP explanation always accompanies score surfaced to counselors
- Score bands use non-clinical language: "check-in suggested," "outreach recommended"
- Bias audit mandatory before any deployment; this card is updated with results
- Finding bias and correcting it is stronger evidence of care than no audit

---

## Files

| File | Purpose |
|---|---|
| `src/risk_classifier.py` | Training, prediction, evaluation, persistence |
| `models/risk_classifier.pkl` | Serialized trained XGBoost + calibration |
| `notebooks/02_feature_engineering.ipynb` | EDA, correlations, confound analysis |
| `notebooks/03_classifier_training.ipynb` | Training, comparison, calibration |
| `notebooks/04_classifier_explainability.ipynb` | Global + per-prediction SHAP |
| `notebooks/05_classifier_bias_audit.ipynb` | Fairness audit and mitigation |
