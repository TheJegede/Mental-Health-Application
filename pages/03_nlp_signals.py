"""
NLP Distress Signals — text submissions with distress scores, LIME highlighting.
Crisis-flagged content shown at top. Staff access only.
"""

import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv



repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))
load_dotenv(repo_root / ".env")

from src.chatbot.safety import CRISIS_FOOTER_MD

st.set_page_config(
    page_title="NLP Distress Signals — Campus Wellness Navigator",
    page_icon="💬",
    layout="wide",
)

st.warning(
    "**Staff access only.** This page contains confidential student submissions. "
    "Not for student access."
)

# ---------------------------------------------------------------------------
# Sample submissions — representative demo texts, not real student data
# ---------------------------------------------------------------------------

SAMPLE_SUBMISSIONS = [
    {
        "id": "SUB-0042",
        "student_id": "STU-11823",
        "source": "Weekly check-in survey",
        "text": (
            "I've been struggling to get out of bed lately. I missed most of my classes "
            "this week and I feel like everything is hopeless. Nothing seems to matter anymore "
            "and I've stopped talking to my friends."
        ),
        "demo_score": 79.0,
        "demo_layer": "transformer",
        "demo_routing": "counselor_review",
    },
    {
        "id": "SUB-0038",
        "student_id": "STU-07441",
        "source": "Weekly check-in survey",
        "text": (
            "Midterms are really tough this time. I'm worried about my grades and "
            "haven't been sleeping well. I keep waking up at 3am thinking about everything "
            "I need to do. It's hard to focus in class."
        ),
        "demo_score": 54.0,
        "demo_layer": "transformer",
        "demo_routing": "counselor_review",
    },
    {
        "id": "SUB-0035",
        "student_id": "STU-19204",
        "source": "Optional reflective journal",
        "text": (
            "I've been having panic attacks before exams. It's been happening more often "
            "and I don't know how to make it stop. My hands shake and I can't breathe "
            "properly during presentations."
        ),
        "demo_score": 66.0,
        "demo_layer": "transformer",
        "demo_routing": "counselor_review",
    },
    {
        "id": "SUB-0031",
        "student_id": "STU-03892",
        "source": "Weekly check-in survey",
        "text": (
            "Things have been rough since the breakup last month. I'm not eating properly "
            "and I've been isolating from my friends. I know I should reach out but "
            "it's hard to make myself do it."
        ),
        "demo_score": 71.0,
        "demo_layer": "transformer",
        "demo_routing": "counselor_review",
    },
    {
        "id": "SUB-0027",
        "student_id": "STU-22017",
        "source": "Weekly check-in survey",
        "text": (
            "Overall doing okay, just dealing with the usual academic stress. "
            "Exams are coming up but I feel prepared. I've been going to the gym "
            "which helps a lot."
        ),
        "demo_score": 19.0,
        "demo_layer": "vader",
        "demo_routing": "no_action",
    },
    {
        "id": "SUB-0025",
        "student_id": "STU-14556",
        "source": "Optional reflective journal",
        "text": (
            "This semester has been challenging but I'm managing. I started seeing "
            "a therapist at CAPS which has been really helpful. Still stressed about "
            "my thesis but feeling more grounded than I was last month."
        ),
        "demo_score": 28.0,
        "demo_layer": "vader",
        "demo_routing": "no_action",
    },
]

# Crisis-routed — shown separately at top (already handled by system)
CRISIS_SUBMISSIONS = [
    {
        "id": "SUB-0044",
        "student_id": "STU-08731",
        "source": "Weekly check-in survey",
        "text": "I don't see the point in continuing. Everything is falling apart and I'm exhausted.",
        "routed_layer": "lexicon",
        "crisis_category": "suicide_ideation",
        "timestamp": "Today, 09:14",
    },
]


def _render_lime_html(text: str, lime_explanation: list[dict]) -> None:
    """Render LIME word highlights as colored HTML spans."""
    weight_map = {item["word"].lower(): item["weight"] for item in lime_explanation}
    words = text.split()
    html_parts = []
    for word in words:
        clean = word.lower().strip(".,!?;:'\"")
        weight = weight_map.get(clean, 0.0)
        if weight > 0.05:
            bg = f"rgba(220,50,50,{min(abs(weight) * 2, 0.7):.2f})"
            html_parts.append(f'<span style="background:{bg}; padding:1px 3px; border-radius:3px">{word}</span>')
        elif weight < -0.05:
            bg = f"rgba(50,100,220,{min(abs(weight) * 2, 0.7):.2f})"
            html_parts.append(f'<span style="background:{bg}; padding:1px 3px; border-radius:3px">{word}</span>')
        else:
            html_parts.append(word)
    st.markdown(" ".join(html_parts), unsafe_allow_html=True)


# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading NLP pipeline...")
def load_nlp():
    try:
        from src.nlp_distress import load_pipeline
        return load_pipeline(use_lime=False)
    except Exception as e:
        return None


nlp = load_nlp()
nlp_available = nlp is not None

if not nlp_available:
    st.info(
        "**Demo mode:** NLP pipeline not loaded. "
        "Scores shown are pre-computed demo values. "
        "Run notebooks/06_nlp_finetuning.ipynb to enable live analysis."
    )

# ---------------------------------------------------------------------------

st.title("NLP Distress Signals")
st.caption("Text submissions from check-in surveys and optional reflective journals.")

# --- Crisis-routed submissions — always first ---
if CRISIS_SUBMISSIONS:
    st.subheader("Crisis-Flagged Submissions")
    st.caption(
        "These submissions triggered crisis routing (Layer 1 crisis lexicon). "
        "A counselor must follow up. Student has already been shown crisis resources."
    )
    for sub in CRISIS_SUBMISSIONS:
        with st.container(border=True):
            st.error(
                f"**CRISIS ROUTING ACTIVATED** — {sub['timestamp']} · "
                f"Layer: {sub['routed_layer']} · Category: {sub['crisis_category']}"
            )
            cols = st.columns([1, 4])
            with cols[0]:
                st.markdown(f"**{sub['student_id']}**")
                st.caption(sub["source"])
                st.caption(f"Submission: {sub['id']}")
            with cols[1]:
                st.markdown(f"*\"{sub['text']}\"*")
            st.markdown(
                "Action: Student shown crisis resources (988, CTL, UCLA CAPS). "
                "**Assign counselor follow-up.**"
            )
    if "crisis_ack" not in st.session_state:
        st.session_state.crisis_ack = set()
    for sub in CRISIS_SUBMISSIONS:
        if sub["id"] not in st.session_state.crisis_ack:
            if st.button(f"Acknowledge {sub['id']}", type="primary"):
                st.session_state.crisis_ack.add(sub["id"])
                st.rerun()
        else:
            st.success(f"{sub['id']} acknowledged.")

st.divider()

# --- Elevated distress submissions ---
st.subheader("Recent Distress Signals")
st.caption(
    "Submissions ordered by distress score. Run LIME analysis to see which phrases drove each score."
)

sort_order = st.radio(
    "Sort by",
    ["Score (high→low)", "Score (low→high)"],
    horizontal=True,
)

submissions = sorted(
    SAMPLE_SUBMISSIONS,
    key=lambda x: x["demo_score"],
    reverse=(sort_order == "Score (high→low)"),
)

score_filter = st.slider("Minimum distress score to show", 0, 100, 30, 5)
submissions = [s for s in submissions if s["demo_score"] >= score_filter]

if not submissions:
    st.info("No submissions above the current threshold.")

for sub in submissions:
    score = sub["demo_score"]
    routing = sub["demo_routing"]

    if score >= 66:
        border_color = "error"
        band = "Elevated — counselor review recommended"
    elif score >= 41:
        border_color = "warning"
        band = "Moderate — check-in suggested"
    else:
        border_color = None
        band = "Low — no specific action indicated"

    with st.container(border=True):
        top_cols = st.columns([1, 3, 1])

        with top_cols[0]:
            st.markdown(f"**{sub['student_id']}**")
            st.caption(sub["source"])
            st.caption(f"Submission: {sub['id']}")

        with top_cols[1]:
            st.markdown(f"*\"{sub['text']}\"*")

        with top_cols[2]:
            st.metric("Distress Score", f"{score:.0f} / 100")
            st.caption(band)
            st.caption(f"Layer: {sub['demo_layer']}")

        # LIME analysis
        lime_key = f"lime_{sub['id']}"
        if st.button("Analyze with LIME", key=lime_key):
            if nlp_available:
                with st.spinner("Running LIME analysis (~10s)..."):
                    try:
                        result = nlp.analyze(sub["text"], compute_lime=True)
                        if result.lime_explanation:
                            st.markdown("**LIME word importance** (red = distress signal, blue = mitigating)")
                            _render_lime_html(sub["text"], result.lime_explanation)
                        else:
                            st.info("LIME explanation unavailable for this text.")
                    except Exception as e:
                        st.error(f"LIME analysis failed: {e}")
            else:
                st.info("NLP pipeline not loaded. Load the pipeline to run live LIME analysis.")

        # Acknowledgment
        ack_key = f"nlp_ack_{sub['id']}"
        if ack_key not in st.session_state:
            st.session_state[ack_key] = False
        if st.session_state[ack_key]:
            st.success("Acknowledged.")
        else:
            if st.button("Mark as reviewed", key=f"ack_btn_{sub['id']}"):
                st.session_state[ack_key] = True
                st.rerun()


st.divider()

# --- Live text analysis tool ---
st.subheader("Analyze a Submission")
st.caption(
    "Paste any submission text to score it with the NLP pipeline. "
    "Crisis keyword check runs first — always."
)

user_text = st.text_area(
    "Submission text",
    placeholder="Paste student submission text here...",
    height=120,
)

col_btn1, col_btn2 = st.columns([1, 5])
with col_btn1:
    analyze_btn = st.button("Analyze", type="primary", disabled=(not user_text.strip()))
with col_btn2:
    use_lime_check = st.checkbox("Include LIME analysis (slower, ~10s)", value=False)

if analyze_btn and user_text.strip():
    if nlp_available:
        with st.spinner("Analyzing..."):
            try:
                result = nlp.analyze(user_text, compute_lime=use_lime_check)

                if result.crisis_detected:
                    st.error(
                        f"**CRISIS SIGNAL DETECTED** — Category: {result.crisis_category}. "
                        "Student must be shown crisis resources immediately."
                    )
                else:
                    sc, la = result.distress_signal_score, result.layer_reached
                    st.metric("Distress Signal Score", f"{sc:.0f} / 100")
                    st.caption(f"Source layer: {la} | VADER compound: {result.vader_compound:.3f}")
                    if result.lime_explanation:
                        st.markdown("**LIME word importance:**")
                        _render_lime_html(user_text, result.lime_explanation)
                    st.info(result.display_text)

            except Exception as e:
                st.error(f"Analysis failed: {e}")
    else:
        # Run crisis check only (always available)
        try:
            from src.crisis_lexicon import check_crisis, load_lexicon
            lex = load_lexicon()
            match = check_crisis(user_text, lexicon=lex)
            if match.matched:
                st.error(
                    f"**CRISIS KEYWORD DETECTED** — Category: {match.category} · "
                    f"Pattern matched: '{match.pattern}'. Route to crisis resources immediately."
                )
            else:
                st.info("No crisis keywords detected. Full NLP analysis requires the pipeline to be loaded.")
        except Exception as e:
            st.error(f"Crisis check failed: {e}")

st.divider()
st.caption(CRISIS_FOOTER_MD)
