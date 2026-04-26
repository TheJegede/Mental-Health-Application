# Dashboard Walkthrough — Campus Wellness Navigator

**Phase:** 5
**Audience:** Counseling center coordinators (staff) + students (chatbot page only)
**Entry point:** `streamlit run streamlit_app.py`

---

## Pages Summary

| Page | File | Audience | Purpose |
|---|---|---|---|
| Home | `streamlit_app.py` | Staff | Navigation hub, ethics statement, system overview |
| Coordinator Overview | `pages/01_overview.py` | Staff only | KPI cards, weekly trends, demographic distribution |
| Behavioral Risk | `pages/02_behavioral.py` | Staff only | Student risk list, SHAP detail, review workflow |
| NLP Distress Signals | `pages/03_nlp_signals.py` | Staff only | Text submissions, LIME highlighting, live analysis |
| Student Chatbot | `pages/04_chatbot.py` | Students | Crisis-aware resource navigator |
| Model Performance & Fairness | `pages/05_fairness.py` | Staff only | Metrics, bias audit, chatbot safety results |

---

## Page Descriptions

### Home (`streamlit_app.py`)

Landing page. Shows:
- Ethics disclaimer (augments judgment, does not replace)
- Navigation cards linking to each page
- "About this system" expander — full architecture and constraint summary
- Crisis resources footer (appears on every page)

---

### Coordinator Overview (`pages/01_overview.py`)

**KPI Cards (top row)**
- Total students in synthetic cohort
- Students with active recommendations (score > 40)
- Priority follow-up count (score 86–100)
- NLP distress signals last 7 days (simulated in demo)
- Crisis routing events last 7 days (simulated in demo)

**Weekly Recommendation Volume** (line chart)
- 12-week simulated trend with realistic end-of-semester rise in weeks 8–12
- Broken down by score band: check-in / outreach / priority

**Score Band Distribution** (bar chart)
- Current cohort distribution across all four bands

**Recommendation Rate by Demographic Group** (tabbed bar charts)
- Race/ethnicity, gender, first-gen status, financial aid status
- Flagged if any group deviates > 5pp from cohort mean — triggers fairness review

---

### Behavioral Risk (`pages/02_behavioral.py`)

**Controls**
- Threshold slider (default 66 — "outreach recommended" floor, reduces alert fatigue)
- Sort options: score, absence streak, self-report
- "Show all bands" toggle

**Student Table**
- Shows up to 100 students above threshold: ID, score, band, absence streak, self-report, financial stress
- Sortable by any column

**Student Detail** (select from dropdown)
- Score metric + score band with icon
- Full feature table: all 8 behavioral signals with direction labels
- "Run SHAP explanation" button (requires trained model)
- "Mark as reviewed" button — tracks reviewed students in session state

---

### NLP Distress Signals (`pages/03_nlp_signals.py`)

**Crisis-Flagged Submissions** (always shown first)
- Red alert banner with routing layer and crisis category
- Student ID, source, submission text
- Timestamp and "Acknowledge" button
- Counselor follow-up prompt

**Recent Distress Signals**
- Score filter slider (default: show ≥ 30)
- Sort by score high→low or low→high
- Each submission card: student ID, source, text excerpt, distress score, band label
- "Analyze with LIME" button — runs LIME if pipeline loaded, shows word-color HTML
- "Mark as reviewed" per submission

**Live Analysis Tool**
- Paste any text to score on-demand
- "Include LIME analysis" checkbox
- Crisis check runs even without full NLP pipeline (crisis lexicon always available)

---

### Student Chatbot (`pages/04_chatbot.py`)

- Warning banner: "resource navigator, not a counselor"
- Crisis resources persistently in sidebar (988, CTL, UCLA CAPS, 911)
- Chat interface with history
- Source citations on every response (from ChromaDB KB)
- Crisis routing alert when any layer fires
- Response latency shown when > 3000ms (deployment gate reminder)

---

### Model Performance & Fairness (`pages/05_fairness.py`)

**Behavioral Classifier tab**
- Metric targets table (F1, AUC, Brier, calibration error)
- Score band distribution chart
- "Last audited" date

**NLP Distress Detector tab**
- 10/10 crisis recall gate table — one row per canonical crisis query
- Additional metrics: diagnosis violations, FP rate, latency
- Populated by `notebooks/07_nlp_evaluation.ipynb`

**Chatbot Safety tab**
- Three deployment gates: crisis recall, diagnosis language, latency
- Four-layer architecture table
- Known limitations list

**Bias Audit tab**
- Behavioral classifier: demographic parity by race/ethnicity, gender, first-gen, international status
- Deviations > 5pp flagged with Fairlearn remediation note
- NLP bias checks: dialectal variation, writing style, gender parity (targets from plan)
- Chatbot adversarial test results table

---

## Running Locally

```bash
# Install dependencies
uv sync

# Set API key
echo "GROQ_API_KEY=your_key_here" > .env

# Launch
uv run streamlit run streamlit_app.py
```

## Deploying to Streamlit Cloud

1. Push repo to GitHub (no `.env` — key goes in Streamlit secrets)
2. Connect repo in Streamlit Cloud dashboard
3. Add `GROQ_API_KEY` to app secrets
4. App will boot and auto-ingest ChromaDB knowledge base on first run

> **Note:** `torch==2.4.1` is ~2GB. If Streamlit Cloud memory limits are exceeded,
> replace local transformer inference with the Hugging Face Inference API
> (see Risk Register in `mental_health_implementation_plan.md`).

---

## Populating TBD Metrics

Run notebooks in order to fill all TBD slots:

| Notebook | Populates |
|---|---|
| `notebooks/03_classifier_training.ipynb` | `models/risk_classifier.pkl`, classifier metrics |
| `notebooks/05_classifier_bias_audit.ipynb` | Bias audit results for behavioral page |
| `notebooks/06_nlp_finetuning.ipynb` | `models/distress_classifier/` |
| `notebooks/07_nlp_evaluation.ipynb` | NLP crisis recall gate, metrics |
| `notebooks/08_nlp_bias_audit.ipynb` | NLP bias audit results |
| `notebooks/09_chatbot_evaluation.ipynb` | `data/chatbot_safety_audit_results.json` |

---

## Ethics Statement

This dashboard is for counseling center coordinators only. Students access only the chatbot.
No page exposes risk scores directly to students. All outputs use:
*distress signal* · *support recommendation* · *check-in suggested* — never clinical labels.

Crisis resources are present on every page.
