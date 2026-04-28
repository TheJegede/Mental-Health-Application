# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Intent

Student Mental Health Early-Warning AI System — proof-of-concept portfolio project. Three coupled modules behind a Streamlit dashboard for **counseling center coordinators** (not students):

1. **Behavioral risk classifier** (`src/risk_classifier.py`) — scikit-learn / XGBoost pipeline with SMOTE + GridSearchCV + Platt calibration. Outputs `support_recommendation_score` 0–100 with top-3 SHAP features. Never a clinical label.
2. **NLP distress detector** (`src/nlp_distress.py`) — `NLPDistressPipeline`: crisis keyword override → VADER pre-filter → fine-tuned DistilBERT (blended 60/40 with VADER score). LIME explanation on every counselor-facing result.
3. **Crisis-aware chatbot** (`src/chatbot/bot.py`) — `CrisisAwareChatbot`: Groq Llama 3.1 8B + ChromaDB RAG. Four independent safety layers run in order: crisis lexicon → NLP classifier → LLM generation (safety system prompt) → output regex filter.

Streamlit entrypoint `streamlit_app.py` with five pages in `pages/` (01–05). Only `pages/04_chatbot.py` is student-facing; all others are counselor-only.

## Code Architecture

```
src/
  crisis_lexicon.py          # load_lexicon(), check_crisis() — shared by NLP + chatbot
  risk_classifier.py         # train(), predict_score(), save_model(), load_model()
  nlp_distress.py            # NLPDistressPipeline, load_pipeline()
  synthetic_data.py          # 30k-row synthetic student wellbeing generator
  corpus.py                  # load_corpus() for RAG ingestion
  chatbot/
    bot.py                   # CrisisAwareChatbot.respond() — main orchestrator
    knowledge_base.py        # KnowledgeBase: ChromaDB at data/vector_db/
    safety.py                # filter_output(), CRISIS_RESOURCE_TEXT constants
    system_prompt.py         # build_system_prompt()

data/
  crisis_keywords.json       # curated crisis lexicon — edit with care
  synthetic/student_wellbeing.csv
  corpus/                    # markdown KB for RAG (crisis/, counseling/, self_help/, etc.)
  vector_db/                 # ChromaDB store (auto-built by KnowledgeBase)

models/
  risk_classifier.pkl        # ModelBundle (calibrated + raw pipeline + metadata)
  distress_classifier/       # fine-tuned DistilBERT artifacts + checkpoints

notebooks/                   # 01–09, ordered by phase (each consumes prior artifacts)
docs/
  ethics_charter.md
  data_dictionary.md
  plans/phase_1_data_and_ethical_setup.md
```

**Cross-module data flow:** synthetic CSV → notebooks 01–03 → `models/risk_classifier.pkl`. DAIC-WOZ fine-tune → notebook 06 → `models/distress_classifier/`. Chatbot loads both trained models + ChromaDB KB.

**Key types:** `ModelBundle` (risk classifier state), `PredictionResult`, `DistressResult`, `ChatResponse`, `CrisisMatch`, `FilterResult`.

## Commands

Stack: **uv + Python 3.11**. `.env` is local-only (gitignored); `.env.example` is the template.

```bash
uv sync                                    # install deps
uv run jupyter lab                         # open notebooks
uv run streamlit run streamlit_app.py      # run dashboard
```

Streamlit Cloud deploy uses `requirements.txt`. Re-export after dep changes:
```bash
uv pip compile pyproject.toml -o requirements.txt
```

Notebook order matters — later phases consume artifacts from earlier ones. Run them in sequence: 01 → 02 → 03 → 04 → 05 → 06 → 07 → 08 → 09. Phase 4 chatbot reuses Phase 3 NLP classifier; Phase 2 classifier consumes Phase 1 synthetic CSV.

Environment variables needed: `GROQ_API_KEY` (required for chatbot). Set in `.env` locally or Streamlit secrets for cloud.

## Non-Negotiable Ethical Constraints

Load-bearing design constraints. Every change must respect them.

- **Never diagnose.** No DSM-5 terminology, no clinical labels in output. Use "distress signal", "support recommendation", "check-in suggested". `safety.py` enforces this with regex on LLM output.
- **Crisis override runs first, always.** In both `nlp_distress.py` and `chatbot/bot.py`, `check_crisis()` executes before all other inference. False positives acceptable; false negatives are catastrophic. Never reorder, gate, or short-circuit it.
- **Reddit mental-health subreddit data is excluded by design** — do not add it as a data source.
- **Synthetic behavioral data is the chosen path, not a shortcut.** Distributions calibrated to Healthy Minds Study / ACHA-NCHA aggregates; `random_state=42` throughout.
- **Demographic features are retained for bias auditing** — `gender`, `race_ethnicity`, `first_gen`, `international_student`, `financial_aid_status` are in `DEMOGRAPHIC_COLS`. Do not strip them.
- **Every prediction surfaces uncertainty + top contributing features.** SHAP for tabular (`predict_score()`), LIME for text (`NLPDistressPipeline._get_lime_explanation()`).
- **Chatbot evaluation gate:** 100% recall on 10 crisis-language test queries (notebook 09) before deploy.
- **Audience split is structural:** risk scores and NLP scores are counselor-only. `pages/04_chatbot.py` is the only student-facing surface. Do not expose scores to students.

When user requests conflict with the above, push back and cite the constraint.

## Score Bands

Both modules use the same band logic:

| Score | Band |
|-------|------|
| 0–40  | baseline / low distress signal |
| 41–65 | check-in suggested / moderate |
| 66–85 | outreach recommended / elevated |
| 86–100 | priority follow-up |

Crisis keyword match always returns score = 100 and `routing_action = "crisis_resources"`.

## Active Plan

Implementation plan: `mental_health_implementation_plan.md`. Phase plan mirror: `docs/plans/phase_1_data_and_ethical_setup.md`. New sessions: re-read the relevant phase section of the plan before coding.

## Working Notes

- `CrisisAwareChatbot` default campus config is UCLA CAPS — override `campus_name`/`campus_phone` constructor args for different deployments.
- `NLPDistressPipeline` falls back to `distilbert-base-uncased-finetuned-sst-2-english` if fine-tuned model not found — logs a warning; SST-2 inverts label 0 (NEGATIVE) as distress probability.
- `torch==2.4.1` is ~2GB. If Streamlit Cloud memory limits hit, switch to Hugging Face Inference API (noted in `requirements.txt`).
- `data/vector_db/` is auto-built by `KnowledgeBase.build()` from `data/corpus/` — commit the corpus markdown, not the vector store.
- `.env.example` is the committed template. Confirm `.env` contents with user before updating `.env.example`.
