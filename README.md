# Student Mental Health Early-Warning AI System

> **This system does not diagnose. It does not replace counselors. It surfaces distress signals and routes to human support.**

![Python 3.11](https://img.shields.io/badge/python-3.11-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/streamlit-1.38.0-FF4B4B?logo=streamlit&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.1.1-orange)
![Groq](https://img.shields.io/badge/Groq-Llama_3.1_8B-blueviolet)
![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5.5-green)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## Ethical Framing — Read This First

Mental health AI is not like any other machine learning project. False positives cause harm. False negatives cause harm. This system was designed from the ground up with those stakes in mind.

**What this system is:**
- A triage support tool for counseling center coordinators
- A resource navigator chatbot for students
- A proof-of-concept portfolio project

**What this system is not:**
- A diagnostic tool
- A replacement for licensed mental health professionals
- A crisis intervention service
- Ready for institutional deployment without IRB approval and clinical advisory review

Every design decision — from data sourcing to output formatting to the order in which safety layers execute — flows from this framing. See [`docs/ethics_charter.md`](docs/ethics_charter.md) for the full binding constraints.

---

## Problem Context

Campus mental health demand has surged across US higher education while counseling staffing has not kept pace. Most institutions cannot hire clinicians fast enough — but they can use technology to triage, route, and proactively surface students who may benefit from support before they reach crisis.

This system integrates three AI components behind a Streamlit dashboard designed for counseling center coordinators:

1. **Behavioral risk classifier** — flags students whose behavioral patterns suggest reduced wellbeing
2. **NLP distress detector** — scores written text for distress signals with per-word explainability
3. **Crisis-aware chatbot** — student-facing resource navigator with defense-in-depth safety architecture

---

## Architecture

```
                    ┌─────────────────────────────────┐
                    │        Counseling Staff          │
                    │   (not student-facing, except    │
                    │        chatbot page)             │
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
                    │       Streamlit Dashboard        │
                    │  ┌───────────────────────────┐   │
                    │  │ Overview | Behavioral     │   │
                    │  │ Signals  | NLP Signals    │   │
                    │  │ Chatbot  | Fairness Audit │   │
                    │  └───────────────────────────┘   │
                    └──┬──────────┬──────────┬─────────┘
                       │          │          │
          ┌────────────▼──┐  ┌────▼───┐  ┌──▼──────────┐
          │  Behavioral   │  │  NLP   │  │  Crisis-    │
          │  Risk         │  │Distress│  │  Aware      │
          │  Classifier   │  │Detector│  │  Chatbot    │
          │  (XGBoost)    │  │(3-layer│  │  (Groq API) │
          │  + SHAP       │  │ + LIME)│  │  + ChromaDB │
          └───────────────┘  └────────┘  └─────────────┘
                       │          │          │
          ┌────────────▼──────────▼──────────▼─────────┐
          │            Crisis Override Layer            │
          │  (runs FIRST, every input, cannot disable)  │
          └─────────────────────────────────────────────┘
```

---

## Safety Architecture

The crisis override layer runs **before** any classification in every pipeline component. This is non-negotiable. See [`docs/ethics_charter.md §4`](docs/ethics_charter.md) for the invariant specification.

| Layer | Location | Action on trigger |
|---|---|---|
| Crisis keyword regex | NLP pipeline + chatbot input | Immediate crisis resource routing |
| NLP distress classifier | Chatbot input | Elevated routing |
| LLM safety system prompt | Chatbot output | Non-diagnosis enforcement |
| Output regex filter | Chatbot output | Block clinical label language |

---

## Data

**Behavioral classifier training data:** Synthetic — 30,000 rows, `random_state=42`, ~18% positive class prevalence. Distributions calibrated to Healthy Minds Study and ACHA-NCHA published aggregates. Synthetic data is the only ethical option here; using real institutional data would require IRB approval and student consent outside the scope of a portfolio project. See [`docs/data_dictionary.md`](docs/data_dictionary.md).

**NLP model training data:** DAIC-WOZ (USC ICT, academic license) + eRisk shared task data. Raw corpora are gitignored.

**Chatbot knowledge base:** 36 markdown documents across 6 topic areas (crisis, counseling, self-help, academic, peer support, eating disorders). Canonical campus: UCLA CAPS. National crisis resources included. See [`data/corpus/_index.md`](data/corpus/_index.md).

---

## Results

Classifier and NLP metrics populate after running notebooks 03–08. Chatbot safety audit completed.

| Module | Metric | Value |
|---|---|---|
| Behavioral classifier | F1 (positive class) | TBD — run notebook 03 |
| Behavioral classifier | ROC-AUC | TBD — run notebook 03 |
| NLP distress detector | Crisis-language recall | TBD — run notebook 07 |
| Chatbot | Crisis routing recall (10 queries) | **10/10** ✓ gate passed |
| Chatbot | Diagnosis language violations | **0** ✓ |
| Chatbot | Mean response latency | ~11.2s (Groq API; above 3s target — see note below) |
| All modules | Bias audit completed | TBD — run notebooks 05, 08 |

> **Latency note:** The ~11.2s mean latency reflects Groq API round-trips during notebook evaluation. Production latency depends on Groq tier and network conditions. If latency is a constraint, consider streaming responses (`stream=True`) in `src/chatbot/bot.py`.

---

## Live Demo

> **Note:** Deploy link goes here after Streamlit Cloud setup.
> See [Setup](#setup) → [Deploying to Streamlit Cloud](#deploying-to-streamlit-cloud) below.

---

## Setup

**Requirements:** Python 3.11, [uv](https://github.com/astral-sh/uv)

```bash
# Install dependencies
uv sync

# Smoke check (Windows wheels)
uv run python -c "import pandas, numpy, sklearn, xgboost, transformers, torch, chromadb, streamlit, groq"

# Generate synthetic data and build Phase 1 artifacts
uv run jupyter lab
# Open notebooks/01_data_acquisition.ipynb and run all cells

# Launch dashboard (Phase 5+)
uv run streamlit run streamlit_app.py
```

**Environment variables:** Copy `.env.example` to `.env` and populate your keys:
```
GROQ_API_KEY=your_key_here
HUGGINGFACEHUB_API_TOKEN=your_token_here
```

**Run notebooks in order** to generate model artifacts and populate TBD metric slots:
```
notebooks/01_data_acquisition.ipynb        → synthetic CSV + corpus index
notebooks/02_feature_engineering.ipynb     → feature analysis
notebooks/03_classifier_training.ipynb     → models/risk_classifier.pkl
notebooks/04_classifier_explainability.ipynb → SHAP global summary
notebooks/05_classifier_bias_audit.ipynb   → fairness audit results
notebooks/06_nlp_finetuning.ipynb          → models/distress_classifier/
notebooks/07_nlp_evaluation.ipynb          → NLP crisis recall gate
notebooks/08_nlp_bias_audit.ipynb          → NLP bias results
notebooks/09_chatbot_evaluation.ipynb      → chatbot safety audit
```

### Deploying to Streamlit Cloud

1. Create a GitHub repo and push this project
2. Connect repo at [share.streamlit.io](https://share.streamlit.io)
3. Set entry point: `streamlit_app.py`
4. Add `GROQ_API_KEY` to app secrets (not in the repo)
5. ChromaDB knowledge base ingests automatically on first boot

> **Memory note:** `torch==2.4.1` is ~2GB. If Streamlit Cloud memory limits are exceeded,
> replace local transformer inference with the Hugging Face Inference API
> (see Risk Register in `mental_health_implementation_plan.md`).

---

## Repository Layout

```
data/
  synthetic/student_wellbeing.csv   # generated
  corpus/                           # chatbot knowledge base
  crisis_keywords.json              # curated crisis lexicon
  external/                         # gitignored (DAIC-WOZ)
  vector_db/                        # gitignored (rebuilt from corpus)
docs/
  ethics_charter.md                 # binding design constraints
  data_dictionary.md
  dashboard_walkthrough.md
  model_cards/
    risk_classifier.md
    nlp_distress.md
    chatbot.md
  chatbot_safety_audit.md
notebooks/01..09_*.ipynb
src/
  synthetic_data.py
  crisis_lexicon.py
  corpus.py
  risk_classifier.py                # Phase 2
  nlp_distress.py                   # Phase 3
  chatbot/                          # Phase 4
streamlit_app.py + pages/           # Phase 5
models/                             # gitignored
```

---

## Limitations

This system would require the following before any real institutional deployment:
- IRB approval
- Clinical advisory board review
- Pilot validation with real counselors
- Ongoing bias monitoring (not a one-time check)
- Legal review of data handling obligations

These are not hypothetical concerns — they are the difference between a responsible system and a harmful one.

---

## Tech Stack

Python 3.11 · scikit-learn · XGBoost · SHAP · LIME · Fairlearn · AIF360 · DistilBERT/RoBERTa · VADER · Groq API (Llama 3.1 8B) · ChromaDB · Streamlit · uv

---

## Model Cards

- [`docs/model_cards/risk_classifier.md`](docs/model_cards/risk_classifier.md) — XGBoost behavioral classifier
- [`docs/model_cards/nlp_distress.md`](docs/model_cards/nlp_distress.md) — Three-layer NLP distress detector
- [`docs/model_cards/chatbot.md`](docs/model_cards/chatbot.md) — Crisis-aware resource chatbot

---

## Phase Progress

- [x] Phase 1 — Data + Ethical Setup
- [x] Phase 2 — Behavioral Risk Classifier
- [x] Phase 3 — NLP Distress Detection
- [x] Phase 4 — Crisis-Aware Chatbot
- [x] Phase 5 — Counselor Dashboard
- [x] Phase 6 — Portfolio Packaging
