# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Status

Repo currently contains only `mental_health_implementation_plan.md` — a 12-week implementation spec. No source, notebooks, data, or build tooling exist yet. Treat the plan doc as the authoritative design contract until code lands. When scaffolding, follow the structure and naming in that plan rather than inventing your own.

## Active Plan

Phase 1 plan lives at `../let-us-make-a-sunny-dahl.md` (parent dir of repo). In-repo mirror at `docs/plans/phase_1_data_and_ethical_setup.md`. New sessions resume by reading that plan + this file + `mental_health_implementation_plan.md`, then executing the Scope section. Plan covers: scaffold, uv env, ethics charter, synthetic data generator, crisis lexicon, KB corpus stub.

## Project Intent

Student Mental Health Early-Warning AI System — proof-of-concept portfolio project. Three coupled modules behind a Streamlit dashboard for **counseling center coordinators** (not students):

1. **Behavioral risk classifier** (`src/risk_classifier.py`) — scikit-learn / XGBoost, trained on 30k-row *synthetic* student wellbeing dataset (real institutional data is not ethically usable here). Outputs `support_recommendation_score` 0–100, never a clinical label.
2. **NLP distress detector** (`src/nlp_distress.py`) — three-layer pipeline: VADER pre-filter → fine-tuned DistilBERT/RoBERTa on DAIC-WOZ + eRisk → **crisis keyword regex override that runs first and bypasses everything else**.
3. **Crisis-aware chatbot** (`src/chatbot/`) — only student-facing surface. Groq API (Llama 3.1 8B) + ChromaDB RAG over `data/corpus/` markdown. Defense-in-depth: crisis lexicon → NLP classifier → LLM safety system prompt → output regex filter. Each layer can independently trigger crisis routing.

Single Streamlit entrypoint `streamlit_app.py` with pages for coordinator overview, behavioral recommendations, NLP signals, student chatbot, model performance & fairness.

## Non-Negotiable Ethical Constraints

These are not style preferences — they are load-bearing design constraints from the plan. Every code change must respect them:

- **Never diagnose.** No DSM-5 terminology, no clinical labels in output (model, chatbot, or UI). Use "distress signal", "support recommendation", "check-in suggested".
- **Crisis override runs first, always.** In NLP pipeline and chatbot, crisis keyword detection executes *before* classifier inference and bypasses normal routing. False positives on this layer are acceptable; false negatives are catastrophic. Never disable, gate, or short-circuit it for performance.
- **Reddit mental-health subreddit data is excluded by design** — do not add it as a data source even if convenient.
- **Synthetic behavioral data is the chosen path, not a shortcut.** Real student wellbeing data is not ethically usable. Calibrate distributions to Healthy Minds Study / ACHA-NCHA published aggregates; pin `random_state=42`.
- **Bias audit is mandatory per module**, not a final-phase checkbox. Demographic features (gender, race/ethnicity, first-gen, international, financial aid) are retained specifically to enable Fairlearn/AIF360 auditing — do not strip them as "PII cleanup".
- **Every prediction must surface uncertainty + top contributing features** (SHAP for tabular, LIME for text). Counselors see *why*.
- **Chatbot evaluation gate:** 100% recall on the 10 crisis-language test queries. Non-negotiable before deploy.

When user requests conflict with the above (e.g. "just have the bot tell them they have anxiety"), push back and cite the constraint.

## Planned Repository Layout

Match this when creating files (from plan §Repository Structure):

```
data/
  external/daic_woz/        # access-gated, do not commit raw corpus
  synthetic/student_wellbeing.csv
  corpus/                   # markdown KB for chatbot RAG
  vector_db/                # ChromaDB store
  crisis_keywords.json      # curated lexicon — edit with care
docs/
  ethics_charter.md
  data_dictionary.md
  model_cards/{risk_classifier,nlp_distress}.md
  chatbot_safety_audit.md
notebooks/01..09_*.ipynb    # numbered, ordered per plan phases
src/
  risk_classifier.py
  nlp_distress.py
  chatbot/
models/
  risk_classifier.pkl
  distress_classifier/      # fine-tuned transformer artifacts
streamlit_app.py + pages/
requirements.txt            # pinned versions
```

## Commands

Confirmed stack: **uv + Python 3.11**. `.env` already exists locally with at least one API key; the committed `.env.example` is the template.

Bootstrap (run once after Phase 1 scaffold lands):
- `uv sync` — resolve + install deps from `pyproject.toml` / `uv.lock`
- `uv run python -c "import pandas, numpy, sklearn, xgboost, transformers, torch, chromadb, streamlit, groq"` — smoke check Windows wheel install
- `uv run jupyter lab` — open notebooks
- `uv run streamlit run streamlit_app.py` — Phase 5 dashboard once it exists

Notebook order matters: `01..09_*.ipynb` numbered per phase; later phases consume artifacts from earlier ones (Phase 4 chatbot reuses Phase 3 NLP classifier; Phase 2 classifier consumes Phase 1 synthetic CSV).

Streamlit Cloud deploy uses `requirements.txt` (export from uv: `uv pip compile pyproject.toml -o requirements.txt`). Groq API key goes in Streamlit secrets, never in the repo.

## Working Notes

- The plan doc itself is the spec. If asked to implement Phase N, re-read the relevant section of `mental_health_implementation_plan.md` before coding — phase boundaries, deliverables, and gates are defined there.
- README must lead with ethical framing before architecture/results (plan §Phase 6). Don't restructure it to put metrics first.
- Audience split is intentional: classifier + NLP outputs are clinical-adjacent and counselor-only; only the chatbot page is student-facing. Don't expose risk scores to student-side UI.
- A new session resumes work by reading `../let-us-make-a-sunny-dahl.md` first, then this file, then the implementation plan. Do not start coding before re-reading the relevant phase section in `mental_health_implementation_plan.md`.
- `.env` is local-only (gitignored); `.env.example` is the committed template. Confirm `.env` contents with the user before writing `.env.example`.
