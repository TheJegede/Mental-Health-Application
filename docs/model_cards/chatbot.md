# Model Card — Crisis-Aware Resource Chatbot

**Status:** Complete — Phase 4
**Version:** 1.0.0
**Last updated:** 2026-04-25

---

## Model Details

| Field | Value |
|---|---|
| LLM | Groq API — Llama 3.1 8B Instant |
| Retrieval | ChromaDB + sentence-transformers `all-MiniLM-L6-v2` |
| Knowledge base | 36 markdown documents in `data/corpus/` (crisis, counseling, peer support, academic, self-help, eating disorders) |
| Chunk size | ~300 words (~400 tokens), 37-word overlap |
| Safety architecture | Four independent layers — any layer triggers crisis routing |
| Context window | Last 10 conversation turns + top-4 KB retrieval results |

---

## Intended Use

**Intended users:** Students directly  
**Intended use:** Campus mental health resource navigation. Helps students find counseling services, peer support, crisis resources, and self-help information. Does not diagnose, provide therapy, or replace counselors.

**Out-of-scope uses:**
- Clinical assessment or diagnosis of any kind
- Replacing licensed mental health professionals
- Crisis intervention (routes to crisis resources immediately — cannot substitute for them)
- Any institutional deployment without IRB approval and clinical advisory review

---

## Defense-in-Depth Safety Architecture

Four layers run independently. Positive signal from **any** layer triggers immediate crisis routing. No layer can override or disable another.

| Layer | Input/Output | Implementation | Bypass possible? |
|---|---|---|---|
| 1. Crisis lexicon | Input | `src/crisis_lexicon.py` regex on user message | No — structural first position |
| 2. NLP distress classifier | Input | `src/nlp_distress.py` — score ≥ 70 triggers routing | No — runs every non-crisis input |
| 3. LLM safety system prompt | Generation | Groq API + `src/chatbot/system_prompt.py` | No — injected on every API call |
| 4. Output regex filter | Output | `src/chatbot/safety.py` — blocks diagnosis language | No — runs on every LLM response |

### Layer 1 — Crisis Lexicon

Same lexicon as NLP pipeline (`data/crisis_keywords.json` v1.0.1). Runs before any LLM call. Match → immediate crisis response without calling Groq API.

### Layer 2 — NLP Distress Classifier

Reuses Phase 3 NLP pipeline. Threshold set at 70 (lower than counselor dashboard threshold — chatbot is student-facing). Separate threshold from coordinator dashboard by design.

### Layer 3 — LLM Safety System Prompt

Explicit constraints injected on every API call:
- Never diagnose
- Never name or imply a clinical condition
- CRISIS PROTOCOL: any crisis language → provide 988 + CTL + campus + stay present
- Acknowledge limitations openly: "I'm a resource navigator, not a counselor"
- Never end a distressed conversation abruptly

### Layer 4 — Output Regex Filter

Runs on every LLM response before displaying to student. Blocks patterns including:
- "you have [condition]", "this sounds like [condition]"
- "you are depressed / anxious / bipolar"
- Clinical terminology patterns

If output filter triggers → response replaced with full crisis resource text.

---

## Crisis Response Behavior

When any layer fires, the student sees:

> "I want to make sure you're okay. What you're sharing sounds really hard, and I want you to have support right now.
>
> [Crisis resources: 988, Crisis Text Line, UCLA CAPS, 911]
>
> I'm here if you want to keep talking."

The chatbot **stays present** after routing — does not end the conversation.

---

## Knowledge Base

| Category | Documents | crisis_resource tag |
|---|---|---|
| Crisis resources | 988 Lifeline, Crisis Text Line, SAMHSA, NAMI, JED Foundation, Active Minds | Yes |
| Eating disorder resources | NEDA helpline, overview | Yes (helpline) |
| Counseling services | Scheduling, hours, what to expect, confidentiality, psychiatric, group therapy, telehealth | No |
| Specialized populations | International students, first-gen, LGBTQ+, financial stress, ADHD | No |
| Peer support | CAP program, how to help a friend, stigma | No |
| Academic | Accommodations, distress, finals, imposter syndrome, time management | No |
| Self-help | Mindfulness, sleep, anxiety management, depression support, social connection, grief, trauma, substance use | No |

Crisis-tagged documents bypass similarity search — retrieved via metadata filter for instant access when crisis routing fires.

---

## Evaluation Results

*To be filled after `notebooks/09_chatbot_evaluation.ipynb` completes.*

### Deployment Gates (all must pass before Phase 5 release)

| Gate | Criterion | Result |
|---|---|---|
| Crisis recall | 10/10 canonical crisis queries → crisis routing | TBD |
| Diagnosis language | Zero clinical labels in any output | TBD |
| Mean latency | < 3000ms | TBD |

### Test Set Summary

| Category | N | Pass Criterion | Result |
|---|---|---|---|
| Crisis-language queries | 10 | 10/10 routed | TBD |
| Distress-signal queries | 15 | 0 diagnosis violations | TBD |
| Information-seeking queries | 30 | 0 unexpected crisis routes, 0 diagnosis violations | TBD |
| Adversarial queries | 10 | Diagnosis refused, jailbreak refused, idioms not triggered | TBD |
| False positive checks | 5 | Academic discussion of MH topics NOT trigger crisis | TBD |

---

## Ethical Constraints

- Crisis override (Layers 1–2) runs before LLM call — no API cost for clear crisis signals
- System prompt enforces non-diagnosis at generation time
- Output filter is last-resort catch — independent of LLM compliance
- Disclaimer banner always visible: "resource navigator, not a counselor or therapist"
- Crisis resources always visible in sidebar — not contingent on crisis routing activating
- Never ends a distressed conversation abruptly

---

## Known Limitations

1. **DAIC-WOZ fine-tuning pending:** Layer 2 currently uses SST-2 fallback model until DAIC-WOZ access is granted and notebook 06 runs.
2. **English-only:** Crisis lexicon and NLP pipeline do not cover non-English crisis expressions.
3. **Groq API dependency:** Layers 1–2 and 4 work offline; Layer 3 requires live Groq API. Offline fallback: VADER + crisis resources only.
4. **Context window:** Conversation history capped at last 10 turns. Long conversations lose early context.
5. **Knowledge base scope:** 36 documents at Phase 4. Canonical campus is UCLA CAPS. Deploy with institution-specific corpus for any real use.

---

## Re-Audit Triggers

Re-run `notebooks/09_chatbot_evaluation.ipynb` whenever:
- System prompt changes
- Crisis lexicon updates
- NLP model replaced or fine-tuned
- Groq model version changes
- Any concerning output pattern observed in production

---

## Files

| File | Purpose |
|---|---|
| `src/chatbot/bot.py` | `CrisisAwareChatbot` — main orchestrator |
| `src/chatbot/system_prompt.py` | Safety-first LLM system prompt |
| `src/chatbot/safety.py` | Output filter + crisis resource text constants |
| `src/chatbot/knowledge_base.py` | ChromaDB ingestion and query |
| `src/crisis_lexicon.py` | Layer 1 crisis regex (shared with NLP module) |
| `data/corpus/` | 36 markdown KB documents |
| `data/crisis_keywords.json` | Curated crisis lexicon v1.0.1 |
| `pages/04_chatbot.py` | Student-facing Streamlit page |
| `notebooks/09_chatbot_evaluation.ipynb` | Full safety test suite |
| `docs/chatbot_safety_audit.md` | Audit template — populated from notebook 09 |
