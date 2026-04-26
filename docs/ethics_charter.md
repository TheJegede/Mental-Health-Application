# Ethics Charter — Student Mental Health Early-Warning AI System

**Version:** 1.0.0
**Date:** 2026-04-25
**Author:** Project maintainer
**Status:** Active — binding on all phases

---

## 1. Six Non-Negotiable Principles

These principles are load-bearing design constraints, not style preferences. Every code change, model decision, and UI copy must respect them.

### 1.1 Augment, Never Replace
Every output flows to a human counselor or trusted campus resource. The system never takes action directly. A recommendation without human review is not a recommendation — it is noise.

### 1.2 Transparent Uncertainty
Every prediction surfaces a confidence band and the top contributing signals that drove it. The system never outputs a binary label ("this student is depressed"). It outputs a graded signal with explanation: "Support recommendation score: 72/100 — top contributing signals: declining LMS engagement (−40% over 3 weeks), missed class streak (5 consecutive)."

### 1.3 Crisis Safety Override — Runs First, Always
Any input containing crisis language immediately routes to crisis resources before any classifier inference runs. This layer cannot be disabled, gated, or short-circuited for performance. False positives on this layer are acceptable. False negatives are catastrophic.

### 1.4 No Clinical Labels
The system uses "distress signal," "support recommendation," and "check-in suggested." It never uses DSM-5 terminology, clinical diagnoses, or language that implies professional clinical assessment. This applies to model output, chatbot responses, UI copy, and code comments.

### 1.5 Demographic Fairness
A bias audit is mandatory for every model, completed before deployment — not after. Demographic features (gender, race/ethnicity, first-gen, international, financial aid status) are retained specifically to enable fairness auditing. They are not stripped as "PII cleanup." Finding and correcting bias is stronger evidence of care than presenting a system with no audit at all.

### 1.6 Data Minimization
Only behavioral signals with documented wellbeing correlation are used. No surveillance data collected for its own sake. Every feature in the dataset has a rationale tied to published wellbeing research.

---

## 2. Scope Statement: What This System Is Not

| This system IS | This system IS NOT |
|---|---|
| A triage and routing support tool for counseling staff | A diagnostic tool |
| A signal surfacing system that flags patterns for human review | A replacement for counselors or therapists |
| A resource navigator for students (chatbot only) | A crisis intervention service |
| A proof-of-concept portfolio project | A clinically validated or IRB-approved instrument |
| A tool that surfaces uncertainty explicitly | A system that produces confident clinical verdicts |

This system requires IRB approval, a clinical advisory board review, and pilot validation before any real institutional deployment. The proof-of-concept designation is not a disclaimer — it is a hard boundary on use.

---

## 3. Audience Split

**Counseling center coordinators** have access to:
- Behavioral risk classifier outputs (support recommendation scores + SHAP explanations)
- NLP distress signal feed (text submissions with LIME-highlighted phrases)
- Model performance and fairness dashboards

**Students** interact only with:
- The crisis-aware chatbot (resource navigator, not clinical tool)

Risk scores and NLP distress signals are never surfaced to students. The chatbot page is the only student-facing surface.

---

## 4. Crisis Override Invariant

The crisis keyword override layer runs **before** classifier inference in both the NLP pipeline and the chatbot. It is implemented as a regex match against a curated lexicon (`data/crisis_keywords.json`).

**Invariants that must hold at all times:**
- The override runs first — no pipeline restructuring may change execution order
- It cannot be disabled by a configuration flag, environment variable, or code path
- It cannot be short-circuited for latency or cost reasons
- Every test suite for the NLP module and chatbot must include the 10 canonical crisis-language test cases
- 100% recall on crisis-language queries is a non-negotiable deployment gate

---

## 5. Data Sourcing Decisions

### 5.1 Reddit Mental Health Data — Excluded by Design
Reddit mental health subreddit data is not used. The population is self-selected for active distress; consent for research use is ambiguous even when posts are technically public; and cleaner alternatives exist. This is not a technical limitation — it is an ethical choice. Do not add Reddit data as a source in any phase.

### 5.2 Synthetic Behavioral Data — The Only Ethical Path
Real student behavioral wellbeing data is not used. Obtaining it would require institutional agreements, IRB approval, and student consent that are outside scope for a portfolio project. Synthetic data calibrated to published national aggregates (Healthy Minds Study, ACHA-NCHA) is the responsible choice, not a shortcut. The README makes this explicit.

**Generation constraints:**
- `random_state=42` for full reproducibility
- Distributions calibrated to published aggregates, not invented
- Positive class prevalence ≈ 18%, matching national moderate-to-severe distress rates
- Demographic features are inputs to behavioral feature distributions, not direct inputs to the label

### 5.3 DAIC-WOZ Corpus
Used under academic research license for NLP fine-tuning only. Raw corpus is gitignored (`data/external/`). Access requires completing the USC ICT request form.

---

## 6. Sunset Criteria

This system should be taken down or suspended if any of the following occur:

1. Bias audit reveals persistent, uncorrectable demographic disparity in false positive or false negative rates across any protected group
2. Crisis override recall drops below 100% on the canonical test set during any re-evaluation
3. Counselors report that system outputs are being used as substitutes for clinical judgment rather than supplements
4. Any student-facing output is found to contain clinical labels, diagnoses, or language that a reasonable person would interpret as professional clinical assessment
5. The proof-of-concept is considered for institutional deployment without prior IRB approval and clinical advisory board review

---

## 7. Pre-Deployment Gates

These gates must all pass before any deployment beyond a portfolio demonstration:

- [ ] 100% recall on the 10 canonical crisis-language test queries
- [ ] Completed bias audit per module (classifier, NLP, chatbot) with mitigation applied and documented
- [ ] IRB approval obtained
- [ ] Clinical advisory board review completed
- [ ] Ethics charter reviewed and updated to reflect actual system behavior
- [ ] Limitations section in README reviewed for accuracy

---

## 8. Sign-Off

| Field | Value |
|---|---|
| Author | Project maintainer |
| Date | 2026-04-25 |
| Version | 1.0.0 |
| Review due | Before any Phase 5 deployment |

This charter binds all phases. Changes require a new version entry in this sign-off block.
