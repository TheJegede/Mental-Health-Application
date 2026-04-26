  
**IMPLEMENTATION PLAN**

**Student Mental Health Early-Warning AI System**

*Proof of Concept — Independent Portfolio Project*

| Timeline | 10–12 Weeks |
| :---- | :---- |
| **Budget** | $0 (open data \+ free tiers) |
| **Modules** | Risk classifier \+ NLP \+ Chatbot |
| **Stack** | Python, Streamlit, Hugging Face, Groq |
| **Risk profile** | High — clinical-adjacent stakes |

# **Ethical Foreword**

Mental health AI is not the same as any prior project in this series. Every prior pain point — retention, enrollment — operated on outcomes that are administrative in nature. This project operates on signals that are clinical-adjacent. False positives cause real harm. False negatives cause real harm. The ethical surface area is large enough that it cannot be a single phase or audit step — it must shape every design decision from the start.

**ETHICAL CONSTRAINT**  This system never diagnoses. It never names a clinical condition. It never replaces a counselor or therapist. It surfaces general distress signals, routes students to existing campus resources, and supports — never substitutes for — professional mental health staff. Every component is designed around this constraint.

## **Six Non-Negotiable Principles**

* Augment, never replace — every output flows to a human counselor or trusted resource, never directly to action

* Transparent uncertainty — every prediction shows confidence and known limitations, never a binary 'this student is depressed'

* Crisis safety override — any input containing crisis language immediately routes to crisis resources, bypassing all classification logic

* No clinical labels — the system uses 'distress signal,' 'support recommendation,' or 'check-in suggested,' never DSM-5 terminology

* Demographic fairness — bias audit on every model, with mitigation applied before deployment, not after

* Data minimization — only behavioral signals that legitimately correlate with wellbeing are used; no surveillance for its own sake

# **Project Overview**

This project addresses the fourth-most-severe pain point in US higher education: the campus mental health crisis. Demand for student counseling services has surged sector-wide, while staffing has not kept pace. Most institutions cannot afford to add clinicians fast enough — but they can use technology to triage, route, and proactively flag students who would benefit from support before they reach crisis.

The system integrates three AI components: a behavioral risk classifier that flags students whose patterns suggest reduced wellbeing, an NLP sentiment/distress detector that scores written content (open-ended survey responses, journal entries) for distress signals, and a crisis-aware chatbot that routes students to appropriate campus resources without ever attempting to diagnose. All three feed a single Streamlit dashboard designed for use by counseling center coordinators — not students directly.

## **Audience Distinction**

Critical design decision up front: this dashboard is for counseling center staff. Students do not have direct access to the classifier outputs. They interact only with the chatbot, which is built on the assumption it may be talking to someone in distress. The classifier and NLP outputs are clinical-adjacent and require human interpretation.

## **Technology Stack**

| Category | Tool / Service | Purpose |
| :---- | :---- | :---- |
| Data manipulation | pandas, numpy | All data processing |
| ML modeling | scikit-learn, XGBoost | Behavioral risk classifier |
| Class imbalance | imbalanced-learn | SMOTE for minority class |
| NLP base model | sentence-transformers, Hugging Face Transformers | Embeddings \+ distress classification |
| Sentiment | VADER, RoBERTa-base sentiment | Multi-method sentiment scoring |
| Explainability | SHAP, LIME for text | Per-prediction feature contribution |
| Fairness | Fairlearn, AIF360 | Bias audit and mitigation |
| Chatbot LLM | Groq API (Llama 3.1 8B) | Free-tier inference for chatbot |
| Crisis detection | Rule-based \+ classifier hybrid | Hard safety override layer |
| Dashboard | Streamlit | Counselor-facing interface |
| Deployment | Streamlit Cloud | Free public hosting |
| Version control | GitHub | Code, notebooks, model artifacts |

| Phase 1  Data Acquisition & Ethical Setup | Weeks 1–2 |
| :---- | :---: |

## **Objective**

Acquire all data sources for the three modules — behavioral data for the classifier, text data for NLP distress detection, and a crisis resource knowledge base for the chatbot. All data must be public, openly licensed, and ethically permissible for mental health research.

**ETHICAL CONSTRAINT**  Reddit mental health subreddit data was deliberately excluded from this project. While public, the population is self-selected for active distress, and using it raises consent questions even when posts are technically public. Cleaner sources exist.

## **Source 1 — DAIC-WOZ Distress Analysis Interview Corpus**

Distress detection text corpus from the University of Southern California's Institute for Creative Technologies. Used originally for clinical depression detection research. Provides labeled text with PHQ-8 scores.

| Attribute | Detail |
| :---- | :---- |
| Source URL | https://dcapswoz.ict.usc.edu/ |
| Access | Free for academic and research use — fill out request form, granted within \~1 week |
| License | Academic research license |
| Size | \~189 interviews, transcribed text |
| Labels | PHQ-8 depression severity score per participant |
| Use in this project | Train NLP distress detector on text segments — model outputs distress signal score, not depression diagnosis |
| Fallback if unavailable | eRisk shared task data, or aggregated public-domain therapy transcripts |

## **Source 2 — eRisk Public Datasets**

Annual shared task on early risk detection on the internet. Provides text samples labeled for early signs of depression and other mental health concerns. Better suited as a supplement or backup to DAIC-WOZ.

* Source: https://erisk.irlab.org/

* License: research-use license, request via shared task organizers

* Use: validation set for the NLP distress detector — confirms the model generalizes beyond a single corpus

## **Source 3 — Synthetic Behavioral Wellbeing Dataset**

Behavioral wellbeing data with mental health outcomes is not publicly available — and would not be ethical to use even if a private institution offered it. The risk classifier therefore trains on synthetic data with realistic distributions calibrated to published academic research on student wellbeing indicators.

* Sample size: 30,000 synthetic students across one academic year

* Behavioral features: LMS engagement variance, sleep schedule estimates from device usage timestamps, social activity flags (group event RSVPs, peer interactions), academic decline trends, missed class patterns, financial stress indicators

* Survey features: simulated 8-item wellbeing self-report scores aggregated across the term

* Calibration sources: published meta-analyses on student wellbeing prevalence (Lipson et al. Healthy Minds Study aggregated statistics, ACHA-NCHA national health assessment summary statistics)

* Target label: binary 'support\_recommended' — calibrated to \~18% prevalence matching national averages of students reporting moderate-to-severe distress

* Demographic features included to enable bias audit: gender, race/ethnicity, first-gen status, international student flag, financial aid status

* Reproducibility: random\_state=42, generation logic fully documented

*Synthetic data here is the only ethical option. The README must explicitly explain this — using fake data is not a shortcut, it is the responsible choice given the population and the stakes.*

## **Source 4 — Campus Resource Knowledge Base**

The chatbot needs a knowledge base of mental health resources to route students to. This is built from publicly available campus counseling center websites, supplemented with national crisis hotline information.

| Attribute | Detail |
| :---- | :---- |
| Source | Public counseling center sites (e.g. Penn State CAPS, UCLA Counseling) — pick one as canonical |
| Crisis lines | 988 Suicide & Crisis Lifeline, Crisis Text Line (HOME to 741741), SAMHSA helpline |
| National resources | JED Foundation, Active Minds, NAMI HelpLine |
| Eating disorder resources | National Alliance for Eating Disorders helpline (verified active resource) |
| Volume | \~150–200 documents covering: crisis support, scheduling counseling appointments, peer support, academic accommodations, group therapy, self-help resources |
| Storage | Markdown files in data/corpus/ — committed to repo with explicit licensing notes |

## **Phase 1 Deliverables**

* data/external/daic\_woz/ — DAIC-WOZ corpus (after access granted)

* data/synthetic/student\_wellbeing.csv — 30,000-row synthetic behavioral dataset

* data/corpus/ — markdown knowledge base for chatbot

* notebooks/01\_data\_acquisition.ipynb — reproducible download \+ synthesis

* docs/data\_dictionary.md — every field documented

* docs/ethics\_charter.md — written ethical principles and design constraints

| Phase 2  Behavioral Risk Classifier | Weeks 2–4 |
| :---- | :---: |

## **Objective**

Build a machine learning classifier that flags students whose behavioral patterns suggest they would benefit from a wellbeing check-in. The classifier outputs a support\_recommendation\_score (0–100) and the top contributing behavioral signals — never a clinical label.

## **Feature Engineering**

Features capture deviation from each student's own baseline, not absolute values. A student who has always been low-engagement should not be flagged the same way as a student who was high-engagement and suddenly dropped.

| Feature | Calculation | Wellbeing rationale |
| :---- | :---- | :---- |
| engagement\_variance | Standard deviation of weekly LMS engagement across last 6 weeks | Erratic patterns can indicate disrupted routine |
| sleep\_schedule\_drift | Variance in median daily activity timestamp across 4 weeks | Sleep disruption is a wellbeing indicator |
| social\_activity\_decline | % drop in group event participation week-over-week | Social withdrawal is a key signal |
| academic\_trend | Linear slope of grades across submissions to date | Academic decline correlates with wellbeing |
| self\_report\_score | Aggregated wellbeing self-report from optional surveys | Direct, consented input from student |
| missed\_class\_streak | Longest consecutive absence streak in past 30 days | Pattern of withdrawal from routine |
| financial\_stress\_flag | Binary: financial aid issue or hold active | Major life stressor |
| help\_seeking\_flag | Binary: visited counseling center or wellness resources | Positive signal — student already engaging |

## **Model Selection & Training**

* Logistic regression as interpretable baseline — coefficients show feature direction directly

* XGBoost as primary challenger — handles non-linear interactions

* Random Forest as secondary — useful for feature importance comparison

* Class balance: \~18% positive class — apply SMOTE on training set only

* Validation: stratified 70/15/15 split, 5-fold cross-validation

* Hyperparameter tuning: GridSearchCV with F1-score on positive class as scoring metric

* Calibration: Platt scaling so the score is a true probability

## **Critical Output Design**

The classifier output is the single most consequential design decision in this project. Wrong framing creates harm even if the underlying model is accurate.

* Output IS: 'Support recommendation score: 72/100 — top contributing signals: declining LMS engagement (-40% over 3 weeks), missed class streak (5 consecutive)'

* Output IS NOT: 'Risk of depression: HIGH' or 'Likely diagnosis: anxiety disorder'

* Score bands: 0–40 baseline, 41–65 check-in suggested, 66–85 outreach recommended, 86–100 priority follow-up

* Every score includes uncertainty: the calibrated probability AND a confidence band based on data sufficiency

* Three contributing factors shown via SHAP — counselor sees what behavioral pattern drove the score

## **Bias Audit — Mandatory and Visible**

Mental health AI has a documented history of bias. Models trained on majority-group data can systematically miss distress signals in minority students or pathologize culturally normal behaviors. Audit is non-negotiable.

| Check | Method |
| :---- | :---- |
| Demographic parity | Compare flag rate across race/ethnicity, gender, first-gen, international status — flag deviation \> 5pp |
| Equalized odds | True positive rate across groups — model must catch distress equally |
| False positive rate parity | Cost of false positive is high (unnecessary outreach can feel intrusive) — must be balanced across groups |
| Cultural normality check | Manually review feature contributions for any pattern that may reflect cultural difference, not distress (e.g., quiet engagement style misread as withdrawal) |
| Mitigation if found | Re-weighting, threshold adjustment per group, or feature exclusion — document before/after |

## **Phase 2 Deliverables**

* src/risk\_classifier.py — training and prediction module

* models/risk\_classifier.pkl — serialized trained model

* notebooks/02\_feature\_engineering.ipynb — feature walkthrough

* notebooks/03\_classifier\_training.ipynb — training, validation, calibration

* notebooks/04\_classifier\_explainability.ipynb — SHAP analysis

* notebooks/05\_classifier\_bias\_audit.ipynb — fairness audit and mitigation

* docs/model\_cards/risk\_classifier.md

| Phase 3  NLP Distress Detection Module | Weeks 4–6 |
| :---- | :---: |

## **Objective**

Build an NLP module that scores written student input — open-ended survey responses, journal entries, optional reflective writing — for distress signals. Like the classifier, it outputs a graded signal, not a diagnosis. Used by counselors as one input among many.

## **Three-Layer Detection Approach**

### **Layer 1 — Lexical Sentiment (Fast Pre-Filter)**

* VADER sentiment analysis — rule-based, fast, picks up obvious negative language

* Used as initial signal — if compound score \> \-0.3, no further processing needed for low-risk content

* Rationale: most text is not distressed; expensive models should only run on potentially-distressed input

### **Layer 2 — Transformer-Based Distress Classifier**

* Base model: distilbert-base-uncased or RoBERTa-base

* Fine-tuning: train on DAIC-WOZ \+ eRisk samples with PHQ-8 score binarized at clinically-meaningful threshold

* Output: distress probability 0–1, mapped to signal score 0–100

* Inference: runs on Hugging Face Inference API free tier or locally on CPU

### **Layer 3 — Crisis Keyword Override**

* Hard rule layer that runs BEFORE the classifier

* Detects explicit crisis language patterns related to self-harm, suicide ideation, or immediate danger

* If triggered, IMMEDIATELY routes to crisis resources — bypasses the entire classification pipeline

* Rule list curated from established mental health screening literature, not synthesized

* False positive on this layer is acceptable — false negative is catastrophic

**ETHICAL CONSTRAINT**  The crisis keyword override is the single most important safety feature in this entire project. It must be tested exhaustively, documented openly, and never disabled. Every test of the system must include crisis-language test cases.

## **Explainability for NLP**

* Use LIME for text explanations — highlights which words drove the distress score

* Counselor sees: original text \+ highlighted phrases \+ signal score

* This lets a counselor see whether the model is responding to genuine distress markers or false positives (e.g., academic essay about a difficult topic)

## **Bias Audit for NLP**

* Same demographic parity checks as the behavioral classifier

* Additional check specific to NLP: dialectal variation — model should not pathologize language patterns common in non-dominant English dialects

* Test set includes deliberately constructed examples in different writing styles to verify equitable handling

## **Phase 3 Deliverables**

* src/nlp\_distress.py — three-layer detection pipeline

* models/distress\_classifier/ — fine-tuned transformer artifacts

* data/crisis\_keywords.json — documented crisis lexicon

* notebooks/06\_nlp\_finetuning.ipynb — training walkthrough

* notebooks/07\_nlp\_evaluation.ipynb — test set scoring including crisis cases

* notebooks/08\_nlp\_bias\_audit.ipynb — fairness across writing styles

* docs/model\_cards/nlp\_distress.md

| Phase 4  Crisis-Aware Resource Chatbot | Weeks 6–9 |
| :---- | :---: |

## **Objective**

Build a chatbot that students interact with directly — but built on the assumption that any conversation could involve distress. The chatbot does not diagnose, does not provide therapy, and does not replace counselors. It listens, validates, and routes to appropriate campus and national resources.

**ETHICAL CONSTRAINT**  This chatbot will be the only component a student interacts with directly. It carries the highest single risk in the project. Every output is filtered through the crisis safety layer before delivery.

## **Architecture — Defense in Depth**

The chatbot uses a layered safety architecture. Each layer runs independently, and a positive signal from any single layer triggers crisis routing immediately.

| Layer | Function | Implementation |
| :---- | :---- | :---- |
| Layer 1: Crisis lexicon | Hard keyword/pattern match | Regex on user input — fastest gate |
| Layer 2: NLP classifier | Distress probability scoring | Reuse Phase 3 classifier |
| Layer 3: LLM safety prompt | System prompt enforces non-diagnosis, crisis-routing | Groq API with strict instructions |
| Layer 4: Output filter | Final regex pass on LLM output before display | Block any diagnosis-like language |

## **Knowledge Base Pipeline**

* Ingest: load all markdown files from data/corpus/

* Chunk: split each document into \~400-token passages with 50-token overlap

* Embed: sentence-transformers all-MiniLM-L6-v2

* Store: ChromaDB at data/vector\_db/

* Crisis resources are pre-tagged in metadata for instant retrieval bypassing similarity search

## **System Prompt Design**

The system prompt is the single most carefully written piece of content in the entire project. Three explicit constraints enforced:

* Never diagnose, name a clinical condition, or suggest one — even when asked directly

* Always default to compassionate listening and resource referral

* If the user mentions self-harm, suicide, or immediate danger — even ambiguously — IMMEDIATELY provide 988 Lifeline \+ Crisis Text Line \+ campus emergency resources, then offer to continue talking

* Acknowledge limitations openly: 'I'm a resource navigator, not a counselor — please reach out to \[resource\] for actual support'

* Never end a distressed conversation abruptly

## **Evaluation Test Set**

The chatbot evaluation set is structured to test the safety architecture, not just the answer quality.

* 30 information-seeking queries (counseling hours, how to schedule, peer support availability)

* 15 distress-signal queries (general anxiety mentions, stress, sleep issues, loneliness)

* 10 crisis-language queries (must trigger crisis routing 100% of the time)

* 10 adversarial queries (attempts to elicit diagnosis, attempts to bypass safety)

* 5 false positive checks (academic discussion of mental health topics — must NOT trigger crisis routing)

* Pass criteria: 100% crisis-language recall, zero diagnosis language in output, \< 3 second average latency

## **Phase 4 Deliverables**

* src/chatbot/ — full multi-layer chatbot module

* data/vector\_db/ — populated knowledge base

* notebooks/09\_chatbot\_evaluation.ipynb — full test set scoring

* docs/chatbot\_safety\_audit.md — every safety test result documented

* Streamlit chatbot page wired to live Groq API with all safety layers active

| Phase 5  Counselor Dashboard & Integration | Weeks 8–11 |
| :---- | :---: |

## **Objective**

Combine all three modules into a single Streamlit application. The dashboard is designed for counseling center coordinators — not for direct student access (except for the chatbot page, which is student-facing).

## **Page Structure**

### **Page 1 — Coordinator Overview**

* KPI cards: students with active support recommendations, NLP distress signals from past 7 days, chatbot conversations, crisis routing events

* Trend chart: support recommendation volume over time

* Demographic distribution of recommendations (triggers visual bias check)

* Quick links to detail pages

### **Page 2 — Behavioral Risk Recommendations**

* Sortable, filterable list of students with active recommendations

* Each card: support score, score band, top 3 contributing signals, recommended action level

* Click-through to per-student detail with SHAP explanation

* Mark-as-reviewed workflow simulation

* Threshold slider — start conservative (\>65) to avoid alert fatigue

### **Page 3 — NLP Distress Signals**

* Recent text submissions with elevated distress scores

* LIME highlighting on each — counselor sees which phrases drove the score

* Crisis-flagged content displayed separately at top, color-coded prominently

* Per-submission acknowledgment workflow

### **Page 4 — Student Chatbot (Public)**

* Streamlit chat interface — the only student-facing page

* Crisis resources displayed persistently in sidebar

* 'Talk to a counselor' button prominently visible at all times

* Disclaimer banner: 'This is a resource navigator, not a counselor or therapist'

* Source citations on every response

### **Page 5 — Model Performance & Fairness**

* Behavioral classifier metrics: F1, recall on positive class, calibration plot

* NLP classifier metrics including 100% crisis-recall verification

* Chatbot safety audit summary

* Bias audit results across demographic groups for ALL three modules

* Visible 'last audited' date for each — fairness is a continuous commitment, not a one-time check

## **Deployment**

* Push to GitHub, connect Streamlit Cloud

* Groq API key stored as Streamlit secret

* Live URL added to README and resume

* Crisis resources displayed in app footer on every page

## **Phase 5 Deliverables**

* streamlit\_app.py — full multi-page application

* requirements.txt with pinned versions

* Live public Streamlit Cloud URL

* docs/dashboard\_walkthrough.md with screenshots

| Phase 6  Portfolio Packaging | Weeks 11–12 |
| :---- | :---: |

## **Objective**

Package the project for portfolio presentation. Given the sensitivity of the domain, the README must lead with ethical framing — not technical specs. Anyone evaluating this work should understand the constraints and care taken before they evaluate the code.

## **Repository Structure**

| Path | Contents |
| :---- | :---- |
| README.md | Ethical framing FIRST, then problem, architecture, results, live link |
| docs/ethics\_charter.md | Six non-negotiable principles, scope boundaries, what this system does NOT do |
| docs/model\_cards/ | Standardized cards for classifier, NLP module, and chatbot |
| docs/chatbot\_safety\_audit.md | Full safety test results |
| notebooks/ | 01-09 — data, classifier, NLP, chatbot, all bias audits |
| src/ | risk\_classifier, nlp\_distress, chatbot/ |
| streamlit\_app.py \+ pages/ | Multi-page dashboard |
| models/ | Serialized artifacts |
| data/ | synthetic, corpus, vector\_db, crisis\_keywords |
| requirements.txt | Pinned dependencies |

## **README Requirements**

* Lead with the ethical framing — what this system is NOT (not a diagnosis tool, not a therapist replacement, not a crisis intervention service)

* Then problem context: campus mental health crisis statistics

* Architecture diagram: three modules \+ safety layers \+ dashboard

* Results table: classifier F1, NLP recall, chatbot crisis-recall (must be 100%), bias audit summary

* Tech stack badges

* Honest limitations section: this would need IRB approval, clinical advisory board, and pilot validation before any institutional deployment

* Live URL \+ setup instructions

## **Optional Article**

A 1,500-word article on this project carries portfolio weight specifically because the domain is hard to do responsibly. Lead with the ethics, not the model F1. Show that the builder understands the difference between 'can' and 'should.'

## **Phase 6 Deliverables**

* Polished GitHub repo with ethics-led README

* All notebooks cleaned

* All model cards finalized

* Ethics charter documented

* (Optional) Published Medium or LinkedIn article

# **Master Timeline**

| Phase | Name | Timeline | Key Deliverable |
| :---- | :---- | :---- | :---- |
| 1 | Data \+ Ethical Setup | Weeks 1–2 | All datasets in repo, ethics charter written |
| 2 | Behavioral Classifier | Weeks 2–4 | Trained model, SHAP, bias audit |
| 3 | NLP Distress Detection | Weeks 4–6 | Three-layer pipeline including crisis override |
| 4 | Crisis-Aware Chatbot | Weeks 6–9 | Defense-in-depth chatbot with full safety audit |
| 5 | Counselor Dashboard | Weeks 8–11 | Live multi-page Streamlit app |
| 6 | Portfolio Packaging | Weeks 11–12 | Ethics-led README, polished repo |

# **Risk Register**

| Risk | Likelihood | Mitigation |
| :---- | :---- | :---- |
| DAIC-WOZ access not granted in time | Low | Use eRisk public data; document fallback transparently |
| Synthetic behavioral data not realistic enough | Medium | Calibrate to published Healthy Minds Study and ACHA-NCHA aggregated statistics; review distributions |
| Chatbot fails crisis recall test | Medium | Iterate keyword list and system prompt until 100% recall achieved on test set; non-negotiable gate to deployment |
| NLP model produces false positives on academic text | High | Adversarial test cases in evaluation; LIME visibility ensures counselor sees what triggered the score |
| Bias audit reveals significant disparity | Medium-High | Apply mitigation, document before/after — finding and correcting bias is stronger evidence than no audit |
| Domain perceived as too sensitive for portfolio | Medium | Lead with ethics framing; honest limitations section; demonstrate awareness rather than overstate capability |
| Streamlit Cloud cannot serve transformer model | Medium | Use Hugging Face Inference API for distress classifier instead of local model |
| Three-module scope causes timeline slippage | High | Cut from Phase 6 polish, never from safety testing or bias audits |

# **Closing Note**

This is the riskiest project in this series. Done well, it demonstrates that the builder understands not just AI techniques, but the ethical boundaries of where AI should be applied — and how to apply it responsibly when the answer is 'yes, but carefully.' Done poorly, it becomes a portfolio liability.

The single most important success criterion is not model F1. It is whether someone reading the README walks away thinking: 'This person took the domain seriously, understood the limits, and built something that respects them.' Every design decision in this plan flows from that goal.