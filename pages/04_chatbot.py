"""
Student-facing chatbot page.

This is the ONLY page students access directly.
All four safety layers active. Crisis resources always visible.
"""

import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))
load_dotenv(repo_root / ".env")

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Campus Wellness Navigator",
    page_icon="🌿",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Disclaimer banner — always visible
# ---------------------------------------------------------------------------

st.warning(
    "**This is a resource navigator, not a counselor or therapist.** "
    "For actual mental health support, please contact UCLA CAPS: **(310) 825-0768**. "
    "If you are in crisis, call or text **988** now."
)

# ---------------------------------------------------------------------------
# Sidebar: crisis resources always present
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## Crisis Resources")
    st.markdown(
        """
**If you are in crisis right now:**

- **Call or text 988** — 24/7, free, confidential
- **Text HOME to 741741** — Crisis Text Line
- **UCLA CAPS after-hours:** (310) 825-0768
- **Campus emergency:** (310) 825-1491
- **Immediate danger:** Call **911**
        """
    )
    st.divider()
    st.markdown("## Talk to a Counselor")
    st.markdown(
        """
**UCLA CAPS:**
(310) 825-0768
Mon–Fri 8am–5pm
After-hours support available 24/7
        """
    )
    st.divider()
    st.caption(
        "Campus Wellness Navigator is a resource navigator, not a clinical service. "
        "It does not diagnose, provide therapy, or replace professional mental health support."
    )

# ---------------------------------------------------------------------------
# Load chatbot (cached)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading resource navigator...")
def load_chatbot():
    """Load KB + NLP pipeline + chatbot. Cached across Streamlit reruns."""
    from src.chatbot.knowledge_base import KnowledgeBase, build_knowledge_base
    from src.nlp_distress import load_pipeline
    from src.chatbot.bot import CrisisAwareChatbot

    try:
        kb = build_knowledge_base()
    except Exception as e:
        st.error(f"Knowledge base unavailable: {e}")
        kb = None

    try:
        nlp = load_pipeline(use_lime=False)
    except Exception as e:
        st.warning(f"NLP layer unavailable (crisis lexicon still active): {e}")
        nlp = None

    api_key = os.environ.get("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
    if not api_key:
        st.error("GROQ_API_KEY not set. Add to .env or Streamlit secrets.")
        return None

    return CrisisAwareChatbot(
        groq_api_key=api_key,
        kb=kb,
        nlp_pipeline=nlp,
    )

chatbot = load_chatbot()

# ---------------------------------------------------------------------------
# Chat interface
# ---------------------------------------------------------------------------

st.title("Campus Wellness Navigator")
st.caption("I'm here to help you find mental health resources and information.")

if "messages" in st.session_state and st.session_state.messages:
    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("Clear chat"):
            st.session_state.messages = []
            st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown(
            "Hi, I'm the Campus Wellness Navigator. I can help you find information about "
            "counseling services, campus resources, and mental health support. "
            "What's on your mind today?"
        )

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources", expanded=False):
                for src in msg["sources"]:
                    title = src.get("metadata", {}).get("title", "Resource")
                    url = src.get("metadata", {}).get("source_url", "")
                    st.markdown(f"- [{title}]({url})" if url else f"- {title}")

if prompt := st.chat_input("Ask me anything about campus mental health resources..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if chatbot is None:
            response_text = (
                "I'm not available right now. For immediate support: "
                "call or text **988**, or text HOME to **741741**."
            )
            st.markdown(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})
        else:
            with st.spinner(""):
                history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages[:-1]
                ]
                response = chatbot.respond(prompt, conversation_history=history)

            st.markdown(response.message)

            if response.crisis_routed:
                st.error(
                    f"Crisis routing activated (layer: {response.crisis_layer}). "
                    "Please reach out to a crisis resource above."
                )

            if response.sources:
                with st.expander("Sources", expanded=False):
                    for src in response.sources:
                        title = src.get("metadata", {}).get("title", "Resource")
                        url = src.get("metadata", {}).get("source_url", "")
                        relevance = src.get("relevance", 0)
                        line = f"- [{title}]({url})" if url else f"- {title}"
                        st.markdown(f"{line} *(relevance: {relevance:.2f})*")

            st.session_state.messages.append({
                "role": "assistant",
                "content": response.message,
                "sources": response.sources,
                "crisis_routed": response.crisis_routed,
            })

            if response.latency_ms > 3000:
                st.caption(f"Response time: {response.latency_ms:.0f}ms (target: <3000ms)")
