# Phase 1 — Data + Ethical Setup (Weeks 1–2)

> **Project:** Student Mental Health Early-Warning AI System
> **Repo:** `C:\Users\jeged\Downloads\Mental-Health-Application\`
> **Spec doc:** `Mental-Health-Application/mental_health_implementation_plan.md`
> **Operating contract:** `Mental-Health-Application/CLAUDE.md`
> **How to resume:** open a new Claude Code session in `Mental-Health-Application/`, point Claude at this file, then begin executing the Scope section.

## Context

Repo currently holds only `mental_health_implementation_plan.md` (the 12-week spec) and `CLAUDE.md` (the operating contract for future Claude sessions). No code, no git, no environment, no data. This plan executes **Phase 1 of the implementation plan** end-to-end so that Phases 2–4 (behavioral classifier, NLP distress detector, crisis-aware chatbot) become unblocked.

Phase 1 is the right unit because every later module consumes a Phase 1 artifact:
- Phase 2 classifier ← `data/synthetic/student_wellbeing.csv`
- Phase 3 NLP override ← `data/crisis_keywords.json`
- Phase 4 chatbot RAG ← `data/corpus/`
- All phases ← `docs/ethics_charter.md` (the design contract)

Domain is clinical-adjacent. Ethical scaffolding lands **before** any model code — this plan is structured to enforce that.

## Confirmed Decisions (from dialogue)

- **Plan scope:** Phase 1 full (scaffold + ethics + synthetic data + crisis lexicon + KB stub).
- **Python tooling:** uv + Python 3.11.
- **`.env`:** already exists locally with at least one API key. New session must confirm contents before writing `.env.example`.
- **Other access state (DAIC-WOZ form, Groq, HF, Streamlit Cloud, GitHub):** not yet confirmed — Claude will prompt as each becomes relevant.

## Scope

In:
1. Repo scaffold + git init + `.gitignore` + `.env.example`
2. uv-managed Python 3.11 env + pinned `requirements.txt`
3. `docs/ethics_charter.md` (six non-negotiable principles formalized)
4. `docs/data_dictionary.md` (every synthetic field documented + calibration sources cited)
5. `docs/model_cards/{risk_classifier,nlp_distress}.md` skeletons (filled in later phases)
6. `data/synthetic/student_wellbeing.csv` — 30,000-row generator, `random_state=42`, ~18% positive class
7. `data/crisis_keywords.json` — curated lexicon w/ cited sources
8. `data/corpus/` — 30–50 markdown KB docs to start (plan target 150–200; rest defers to Phase 4)
9. `notebooks/01_data_acquisition.ipynb` — reproducible glue: download/synthesize, write CSV, write KB index
10. `README.md` — ethics-led skeleton (sections to be filled phase-by-phase)

Out (deferred):
- Vector DB build (`data/vector_db/`) → Phase 4
- DAIC-WOZ corpus ingestion (gated on access form turnaround) → Phase 3
- Any model training, SHAP, LIME, fairness audit code → Phases 2–3
- Streamlit app code → Phase 5

## Prerequisites the User Must Action

Claude will pause and prompt before each. None blocks starting Phase 1 work.

| # | Action | When needed | Notes |
|---|---|---|---|
| 1 | Confirm `.env` contents — which API key is already there? (Groq / HF / both?) | Before `.env.example` written | `.env.example` is committed template; real `.env` never commits. |
| 2 | Submit DAIC-WOZ access form at https://dcapswoz.ict.usc.edu/ | Now (≈1 week turnaround) | Phase 3 blocker. File in parallel with Phase 1–2 work. |
| 3 | Create empty GitHub repo + provide remote URL | After local scaffold | Claude will set the remote when given the URL. |
| 4 | Decide canonical campus for KB corpus | Before corpus build | Default: **UCLA Counseling & Psychological Services**. Override if desired. |
| 5 | Hugging Face account + token (free) | Phase 3 | Not needed for Phase 1; flagged early. |
| 6 | Streamlit Cloud account (free) | Phase 5 | Not needed for Phase 1. |

## Repo Layout to Create

```
.gitignore
.env.example
README.md
pyproject.toml          # uv-managed
uv.lock                 # generated
requirements.txt        # exported from uv for Streamlit Cloud compat
docs/
  ethics_charter.md
  data_dictionary.md
  model_cards/
    risk_classifier.md
    nlp_distress.md
  plans/
    phase_1_data_and_ethical_setup.md   # in-repo copy of this plan
data/
  synthetic/
    student_wellbeing.csv          # generated
  corpus/
    _index.md
    crisis/
      988_lifeline.md
      crisis_text_line.md
      samhsa_helpline.md
    counseling/
      scheduling.md
      hours.md
      ...
    peer_support/
    academic/
    self_help/
    eating_disorders/
  crisis_keywords.json
  external/                        # gitignored (raw DAIC-WOZ once granted)
notebooks/
  01_data_acquisition.ipynb
src/
  __init__.py
  synthetic_data.py                # generator module
  crisis_lexicon.py                # loader + match utility (used by Phases 3 & 4)
  corpus.py                        # corpus loader (used by Phase 4)
```

## Critical File Designs

### `pyproject.toml` / `requirements.txt` (uv, Python 3.11)

Pinned set covering all 6 phases so we lock once. Heavy ML stack on Windows — version pins matter. Proposed top-level pins:

```
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.5.1
xgboost==2.1.1
imbalanced-learn==0.12.3
sentence-transformers==3.0.1
transformers==4.44.2
torch==2.4.1                       # CPU wheels on Windows
vaderSentiment==3.3.2
shap==0.46.0
lime==0.2.0.1
fairlearn==0.10.0
aif360==0.6.1
chromadb==0.5.5
streamlit==1.38.0
groq==0.11.0
python-dotenv==1.0.1
jupyterlab==4.2.5
matplotlib==3.9.2
seaborn==0.13.2
```

Phase 1 only strictly needs: pandas, numpy, python-dotenv, jupyterlab, matplotlib. Rest installed up-front to fail fast on Windows wheel issues.

### `.gitignore` essentials

```
.env
.venv/
__pycache__/
*.pyc
.ipynb_checkpoints/
data/external/         # raw third-party corpora — never commit
data/vector_db/        # rebuild from corpus
models/                # train artifacts, large
.DS_Store
```

### `.env.example`

```
GROQ_API_KEY=
HUGGINGFACEHUB_API_TOKEN=
```

Real `.env` already exists locally (user-confirmed). `.env.example` is the committed template.

### `docs/ethics_charter.md`

Sections (all from plan §"Ethical Foreword" — formalize, don't invent):
1. Six non-negotiable principles (verbatim from plan)
2. Scope statement: what this system **is not** (not diagnosis, not therapy, not crisis intervention service)
3. Audience split: counselor-only vs student-facing surfaces
4. Crisis override invariant: "runs first, never disabled, false positives acceptable"
5. Data sourcing decisions:
   - Reddit mental-health data **excluded** (consent + self-selection bias)
   - Synthetic behavioral data is the **only ethical path**, not a shortcut
6. Sunset criteria: conditions under which the system should be taken down
7. Pre-deployment gates (carried forward to Phases 5–6): 100% chatbot crisis-recall, completed bias audit per module, IRB + clinical advisory review before any institutional pilot
8. Sign-off block: author, date, version

### `docs/data_dictionary.md`

For each synthetic column: name, dtype, range/values, generation method, calibration source (Healthy Minds Study / ACHA-NCHA / IPEDS), rationale for inclusion. Plus a top-level "How the label is derived" section because that decides whether the bias audit later is meaningful.

### `data/crisis_keywords.json`

Schema:
```json
{
  "version": "1.0.0",
  "last_reviewed": "YYYY-MM-DD",
  "sources": [
    {"name": "Columbia Suicide Severity Rating Scale (C-SSRS)", "url": "...", "license": "..."},
    {"name": "Coppersmith et al., CLPsych shared task lexicon", "citation": "..."}
  ],
  "categories": {
    "suicide_ideation": {"patterns": ["..."], "examples": ["..."]},
    "self_harm": {"patterns": ["..."], "examples": ["..."]},
    "immediate_danger": {"patterns": ["..."], "examples": ["..."]}
  },
  "review_log": [{"date": "...", "reviewer": "...", "change": "..."}]
}
```

**Critical**: per the implementation plan, "rule list curated from established mental health screening literature, not synthesized." Populate from cited sources (C-SSRS phrasings, published self-harm lexicons), not from model-generated suggestions. Any pattern whose source cannot be verified must be flagged for user review before commit, not silently included.

### `src/synthetic_data.py` — generator design

Function: `generate(n=30_000, seed=42, target_prevalence=0.18) -> pd.DataFrame`

Generation order matters for ethical defensibility:

1. **Demographics first**, sampled from IPEDS-aligned distributions (gender, race/ethnicity, first-gen, international, financial aid). These are **inputs to behavioral feature distributions, not direct inputs to the label**.
2. **Behavioral features**, with realistic correlations:
   - `financial_stress_flag` Bernoulli with rate that depends on `financial_aid_status` (realistic confound)
   - `engagement_variance`, `sleep_schedule_drift`, `social_activity_decline`, `academic_trend`, `missed_class_streak` drawn from distributions calibrated to ACHA-NCHA aggregated stats
   - `self_report_score` (8-item wellbeing) with negative correlation to behavioral risk indicators
   - `help_seeking_flag` Bernoulli; positive signal (already engaging support)
3. **Latent wellbeing score** = weighted sum of behavioral features + Gaussian noise. Demographics do **not** appear directly.
4. **Label** `support_recommended` = 1 where latent score > threshold; threshold tuned via bisection until empirical prevalence ≈ 18%.
5. Persist to `data/synthetic/student_wellbeing.csv`. Document feature weights, threshold, and seed in dataset header comment + data dictionary.

This design lets Phase 2 bias audit reveal real disparity from realistic confounders (e.g., first-gen students disproportionately financial-stressed → flagged more) rather than disparity we hard-coded. That's what makes the audit credible.

### `data/corpus/` — initial KB

~30–50 markdown files at this stage. Each file front-matter:

```markdown
---
title: ...
category: crisis | counseling | peer_support | academic | self_help | eating_disorders
source_url: ...
license: ...
last_verified: YYYY-MM-DD
crisis_resource: true|false
---

# ...content...
```

Crisis resources tagged `crisis_resource: true` so Phase 4 chatbot can do metadata-filtered retrieval (bypass similarity search for crisis routing — required by plan §"Knowledge Base Pipeline").

Initial content:
- 988 Suicide & Crisis Lifeline
- Crisis Text Line (HOME → 741741)
- SAMHSA National Helpline
- National Alliance for Eating Disorders
- NAMI HelpLine
- JED Foundation
- Active Minds
- ~20–40 docs from canonical campus (default UCLA CAPS) covering scheduling, hours, peer support, group therapy, academic accommodations, self-help

### `notebooks/01_data_acquisition.ipynb`

Reproducible glue. Cells:
1. Verify env (Python version, package versions)
2. Call `src.synthetic_data.generate(...)` → write CSV → print summary stats + class balance
3. Plot demographic distributions vs calibration targets (visual sanity check)
4. Validate `data/crisis_keywords.json` schema + count patterns per category
5. Walk `data/corpus/` → emit `_index.md` summary
6. Print "Phase 1 ready ✓" gate w/ checklist

## Critical Files Modified / Referenced

- `mental_health_implementation_plan.md` — read-only reference (the spec)
- `CLAUDE.md` — read-only reference (operating contract)
- All files listed in **Repo Layout** above are new

## Reuse Notes

Nothing pre-exists in the repo to reuse. All new code. External libs reused per `requirements.txt` — no rolling-our-own SMOTE/SHAP/embeddings.

## Verification

End-to-end Phase 1 gate. All must pass before declaring Phase 1 complete:

1. `uv sync` succeeds; `python -c "import pandas, numpy, sklearn, xgboost, transformers, torch, chromadb, streamlit, groq"` exits 0.
2. `jupyter lab` opens; running `notebooks/01_data_acquisition.ipynb` top-to-bottom completes without error.
3. `data/synthetic/student_wellbeing.csv` exists, has 30,000 rows, label prevalence in [0.17, 0.19], all documented columns present.
4. `python -c "import json; d=json.load(open('data/crisis_keywords.json')); assert d['sources'] and all(c['patterns'] for c in d['categories'].values())"` passes.
5. Every file under `data/corpus/` parses as valid markdown with required frontmatter (`title`, `category`, `source_url`, `last_verified`).
6. `git status` shows `.env` ignored, `data/external/` ignored.
7. Manual review: `docs/ethics_charter.md` covers all six principles + audience split + crisis override invariant.
8. README renders w/ ethics framing as the first content section.

## Open Items to Surface During Execution

- Confirm `.env` contents to write matching `.env.example` (will prompt before writing).
- Confirm canonical campus for KB if not UCLA CAPS (will prompt before corpus population).
- Confirm GitHub remote URL when ready to push (will prompt before `git remote add`).
- Any crisis keyword whose source citation cannot be verified will be flagged for user review before commit, not silently included.

## How to Resume in a New Session

1. Open a new Claude Code session in `C:\Users\jeged\Downloads\Mental-Health-Application\`.
2. Tell Claude: "Read `../let-us-make-a-sunny-dahl.md` and continue Phase 1 execution from where it left off. Honor `CLAUDE.md` ethical constraints."
3. Claude reads this plan + CLAUDE.md + the implementation plan and resumes — prompting for the prerequisites above as it hits them.
