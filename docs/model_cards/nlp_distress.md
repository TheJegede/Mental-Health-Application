# Model Card — NLP Distress Detection Module

**Status:** Complete — Phase 3
**Version:** 1.0.0
**Last updated:** 2026-04-25

---

## Model Details

| Field | Value |
|---|---|
| Architecture | Three-layer pipeline: Crisis Override → VADER pre-filter → Transformer |
| Base model | DistilBERT-base-uncased (fine-tuned) |
| Fine-tuning data | DAIC-WOZ + eRisk; PHQ-8 binarized at threshold ≥ 10 |
| Task | Distress signal detection from free-text student input |
| Output | `distress_signal_score` 0–100 + LIME word-level explanation |
| Crisis layer | Regex against `data/crisis_keywords.json` — executes FIRST, always |
| Fallback | `distilbert-base-uncased-finetuned-sst-2-english` if fine-tuned model absent |

---

## Intended Use

**Intended users:** Counseling center coordinators  
**Intended use:** Score written student input (open-ended surveys, journal entries) for distress signals. One input among many; requires human interpretation.

**Out-of-scope uses:**
- Real-time monitoring of student communications
- Automated action without counselor review
- Direct student-facing output (except crisis routing → resources, not scores)
- Any institutional deployment without IRB approval

---

## Execution Order (Non-Negotiable)

```
Input text
    │
    ▼
1. Crisis keyword override  ←── ALWAYS FIRST, cannot be reordered
    │ matched → route to crisis resources immediately (stop)
    ▼
2. VADER compound score
    │ compound > -0.3 → low risk, score from VADER (capped at 40), stop
    ▼
3. Transformer distress classifier
    │ probability × 100 blended with VADER (60/40)
    ▼
Output: distress_signal_score 0–100 + LIME explanation
```

---

## Three-Layer Design

### Layer 1 — Crisis Keyword Override
- Source: `src/crisis_lexicon.py` + `data/crisis_keywords.json` (v1.0.1)
- Runs before VADER and transformer
- Match → score set to 100, routing_action = "crisis_resources"
- False positives acceptable; false negatives are catastrophic
- **10/10 canonical crisis-case recall is a deployment gate**

### Layer 2 — VADER Pre-Filter
- `vaderSentiment.SentimentIntensityAnalyzer`
- Compound threshold: `-0.3` (configurable)
- Text with compound > −0.3 is low risk — skips transformer
- Score capped at 40 for VADER-filtered text

### Layer 3 — Transformer Distress Classifier
- Fine-tuned DistilBERT on DAIC-WOZ + eRisk
- PHQ-8 binary threshold: 10 (moderate-to-severe)
- Final score = 0.6 × transformer_prob × 100 + 0.4 × vader_derived_score
- LIME explanation computed on demand

---

## Distress Score Interpretation

| Score | Band | Action |
|---|---|---|
| 100 | Crisis detected | Immediate crisis resources — do not delay |
| 66–99 | Elevated distress signal | Counselor review recommended |
| 41–65 | Moderate distress signal | Check-in suggested |
| 0–40 | Low distress signal | No specific action indicated |

Output never contains clinical labels or DSM-5 terminology.

---

## Evaluation Results

*To be filled after `notebooks/07_nlp_evaluation.ipynb` completes.*

| Metric | Value |
|---|---|
| Crisis recall (10-case canonical set) | **Must be 10/10** — deployment gate |
| Distress case mean score | TBD |
| False positive triggers (academic text) | TBD |
| Model source | TBD (fine-tuned / fallback) |

---

## Bias Audit Results

*To be filled after `notebooks/08_nlp_bias_audit.ipynb` completes.*

| Check | Result |
|---|---|
| Dialectal variation (mean score divergence) | TBD (target: < 15 pts) |
| Writing style consistency | TBD |
| Gender language parity | TBD |

**Key limitation:** DAIC-WOZ is majority-English clinical interviews. Training data demographic representation must be reviewed before institutional deployment.

---

## Ethical Constraints

- Crisis override cannot be disabled, reordered, or gated
- Output never contains clinical diagnosis language
- LIME explanation always accompanies scores surfaced to counselors
- 10/10 crisis recall is a hard deployment gate (also enforced in notebook 09)

---

## Files

| File | Purpose |
|---|---|
| `src/nlp_distress.py` | Pipeline class, analysis, LIME |
| `src/crisis_lexicon.py` | Crisis override (Phase 1) |
| `data/crisis_keywords.json` | Curated lexicon v1.0.1 |
| `models/distress_classifier/` | Fine-tuned transformer artifacts |
| `notebooks/06_nlp_finetuning.ipynb` | DAIC-WOZ + eRisk fine-tuning |
| `notebooks/07_nlp_evaluation.ipynb` | Crisis recall gate + full eval |
| `notebooks/08_nlp_bias_audit.ipynb` | Dialectal variation + fairness |
