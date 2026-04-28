"""
CrisisAwareChatbot — Phase 4 main orchestrator.

Defense-in-depth: each layer runs independently.
Positive signal from ANY layer triggers crisis routing immediately.

Layer execution order:
  1. Crisis lexicon  (input)   — src.crisis_lexicon.check_crisis()
  2. NLP classifier  (input)   — src.nlp_distress.NLPDistressPipeline
  3. RAG retrieval             — KnowledgeBase.query()
  4. LLM generation            — Groq API + system prompt
  5. Output filter   (output)  — safety.filter_output()
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.chatbot.knowledge_base import KnowledgeBase
from groq import Groq

from src.chatbot.safety import (
    CRISIS_RESOURCE_TEXT,
    CRISIS_RESOURCE_TEXT_SHORT,
    FilterResult,
    contains_crisis_output,
    filter_output,
)
from src.chatbot.system_prompt import build_system_prompt
from src.crisis_lexicon import check_crisis, load_lexicon

_DEFAULT_LEXICON_PATH = Path(__file__).parent.parent.parent / "data" / "crisis_keywords.json"

# NLP distress threshold for chatbot routing (lower than counselor threshold — chatbot is student-facing)
_NLP_CRISIS_THRESHOLD = 70.0


@dataclass
class ChatResponse:
    message: str                        # final message shown to student
    crisis_routed: bool                 # True if any layer triggered crisis routing
    crisis_layer: Optional[str]         # which layer triggered: "lexicon" | "nlp" | "llm_output"
    distress_signal_score: Optional[float]
    sources: list[dict]                 # KB docs cited
    latency_ms: float
    filter_result: Optional[FilterResult]
    raw_llm_response: Optional[str]     # pre-filter LLM output (for audit logging)


class CrisisAwareChatbot:
    """
    Crisis-aware resource chatbot.

    All four safety layers active. Crisis routing triggers on any layer.
    Never diagnoses. Never ends a distressed conversation abruptly.
    """

    def __init__(
        self,
        groq_api_key: Optional[str] = None,
        kb: Optional[KnowledgeBase] = None,
        nlp_pipeline=None,
        model: str = "llama-3.1-8b-instant",
        campus_name: str = "UCLA CAPS",
        campus_phone: str = "(310) 825-0768",
        lexicon_path: str | Path = _DEFAULT_LEXICON_PATH,
        max_tokens: int = 512,
        temperature: float = 0.4,
    ):
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._system_prompt = build_system_prompt(campus_name, campus_phone)
        self._lexicon = load_lexicon(lexicon_path)
        self._kb = kb
        self._nlp = nlp_pipeline

        # Groq client
        api_key = groq_api_key or os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not set. Add to .env or pass groq_api_key= directly."
            )
        self._groq = Groq(api_key=api_key)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def respond(
        self,
        user_message: str,
        conversation_history: list[dict] | None = None,
    ) -> ChatResponse:
        """
        Generate a response to user_message.

        conversation_history: list of {"role": "user"|"assistant", "content": str}
        Returns ChatResponse with final message and full audit metadata.
        """
        t0 = time.time()

        # ----------------------------------------------------------------
        # LAYER 1: Crisis lexicon — RUNS FIRST
        # ----------------------------------------------------------------
        crisis_match = check_crisis(user_message, lexicon=self._lexicon)
        if crisis_match.matched:
            return self._crisis_response(
                user_message,
                crisis_layer="lexicon",
                latency_ms=(time.time() - t0) * 1000,
            )

        # ----------------------------------------------------------------
        # LAYER 2: NLP distress classifier
        # ----------------------------------------------------------------
        distress_score = None
        if self._nlp is not None:
            try:
                nlp_result = self._nlp.analyze(user_message, compute_lime=False)
                distress_score = nlp_result.distress_signal_score
                if nlp_result.crisis_detected or distress_score >= _NLP_CRISIS_THRESHOLD:
                    return self._crisis_response(
                        user_message,
                        crisis_layer="nlp",
                        distress_score=distress_score,
                        latency_ms=(time.time() - t0) * 1000,
                    )
            except Exception as e:
                print(f"[chatbot] NLP layer error (non-fatal): {e}")

        # ----------------------------------------------------------------
        # LAYER 3: RAG retrieval
        # ----------------------------------------------------------------
        sources = []
        context_text = ""
        if self._kb is not None:
            try:
                retrieved = self._kb.query(user_message, n_results=4)
                sources = retrieved
                context_text = self._format_context(retrieved)
            except Exception as e:
                print(f"[chatbot] KB retrieval error (non-fatal): {e}")

        # ----------------------------------------------------------------
        # LAYER 4 (part A): LLM generation with safety system prompt
        # ----------------------------------------------------------------
        messages = self._build_messages(
            user_message,
            conversation_history or [],
            context_text,
        )

        try:
            completion = self._groq.chat.completions.create(
                model=self._model,
                messages=messages,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            )
            raw_response = completion.choices[0].message.content
        except Exception as e:
            raw_response = (
                "I'm having trouble connecting right now. If you need support, "
                f"please reach out directly: {CRISIS_RESOURCE_TEXT_SHORT}"
            )
            print(f"[chatbot] Groq API error: {e}")

        # ----------------------------------------------------------------
        # LAYER 4 (part B): Output filter — runs on every LLM response
        # ----------------------------------------------------------------
        filter_result = filter_output(raw_response)
        final_message = filter_result.filtered

        if filter_result.was_modified:
            print(f"[chatbot] Output filter triggered: {filter_result.triggered_patterns}")
            return self._crisis_response(
                user_message,
                crisis_layer="llm_output",
                distress_score=distress_score,
                latency_ms=(time.time() - t0) * 1000,
                raw_llm=raw_response,
                filter_result=filter_result,
            )

        latency_ms = (time.time() - t0) * 1000
        return ChatResponse(
            message=final_message,
            crisis_routed=False,
            crisis_layer=None,
            distress_signal_score=distress_score,
            sources=sources,
            latency_ms=latency_ms,
            filter_result=filter_result,
            raw_llm_response=raw_response,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _crisis_response(
        self,
        user_message: str,
        crisis_layer: str,
        distress_score: Optional[float] = None,
        latency_ms: float = 0.0,
        raw_llm: Optional[str] = None,
        filter_result: Optional[FilterResult] = None,
    ) -> ChatResponse:
        """Build crisis routing response. Provides all crisis resources + stays present."""
        message = (
            "I want to make sure you're okay. What you're sharing sounds really hard, "
            "and I want you to have support right now.\n\n"
            + CRISIS_RESOURCE_TEXT
            + "\n\nI'm here if you want to keep talking."
        )
        return ChatResponse(
            message=message,
            crisis_routed=True,
            crisis_layer=crisis_layer,
            distress_signal_score=distress_score,
            sources=[],
            latency_ms=latency_ms,
            filter_result=filter_result,
            raw_llm_response=raw_llm,
        )

    def _format_context(self, retrieved_docs: list[dict]) -> str:
        if not retrieved_docs:
            return ""
        parts = ["Relevant campus resources and information:"]
        for doc in retrieved_docs:
            title = doc["metadata"].get("title", "Resource")
            parts.append(f"\n[{title}]\n{doc['text'][:600]}")
        return "\n".join(parts)

    def _build_messages(
        self,
        user_message: str,
        history: list[dict],
        context: str,
    ) -> list[dict]:
        messages = [{"role": "system", "content": self._system_prompt}]

        # Inject KB context as a system addendum (not visible as user message)
        if context:
            messages.append({
                "role": "system",
                "content": f"Context from the campus resource knowledge base:\n{context}",
            })

        # Prior conversation (last 10 turns to stay within context window)
        for turn in history[-10:]:
            messages.append({"role": turn["role"], "content": turn["content"]})

        messages.append({"role": "user", "content": user_message})
        return messages


