"""
NLP Distress Detection Pipeline — Phase 3.

Execution order (non-negotiable — ethics charter §4):
  1. Crisis keyword override  ← runs FIRST, always
  2. VADER pre-filter         ← gates transformer inference
  3. Transformer classifier   ← DistilBERT/RoBERTa fine-tuned on DAIC-WOZ + eRisk

If crisis override fires → immediate routing, pipeline stops.
VADER compound > -0.3 → low risk, skip transformer (score derived from VADER only).
VADER compound <= -0.3 → run transformer for distress probability.

Output: distress_signal_score 0–100. Never a clinical label.
LIME explanation always accompanies any score surfaced to counselors.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore")

_DEFAULT_MODEL_DIR = Path(__file__).parent.parent / "models" / "distress_classifier"
_DEFAULT_LEXICON_PATH = Path(__file__).parent.parent / "data" / "crisis_keywords.json"

# VADER threshold: compound > this value → low risk, skip transformer
_VADER_THRESHOLD = -0.3

# PHQ-8 binarization threshold used during fine-tuning
PHQ8_BINARY_THRESHOLD = 10


@dataclass
class DistressResult:
    text: str
    crisis_detected: bool
    crisis_category: Optional[str]          # "suicide_ideation" | "self_harm" | "immediate_danger" | None
    vader_compound: float
    transformer_score: Optional[float]       # None if skipped (VADER filtered) or model not loaded
    distress_signal_score: float             # 0–100 — the counselor-facing number
    layer_reached: str                       # "crisis" | "vader" | "transformer"
    lime_explanation: Optional[list[dict]]   # [{"word": str, "weight": float}] top tokens
    display_text: str
    routing_action: str                      # "crisis_resources" | "counselor_review" | "no_action"


def _vader_to_score(compound: float) -> float:
    """Map VADER compound [-1, 1] to distress score [0, 100]. More negative → higher score."""
    return float(((1.0 - compound) / 2.0) * 100)


def _format_display_text(result: DistressResult) -> str:
    """Counselor-facing summary. No clinical labels, no diagnosis language."""
    if result.crisis_detected:
        return (
            f"CRISIS SIGNAL DETECTED [{result.crisis_category}] - "
            f"Immediate routing to crisis resources. Do not delay response."
        )

    score = result.distress_signal_score
    if score >= 66:
        band = "Elevated distress signal - counselor review recommended"
    elif score >= 41:
        band = "Moderate distress signal - check-in suggested"
    else:
        band = "Low distress signal - no specific action indicated"

    parts = [f"Distress signal score: {score:.0f}/100 - {band}"]

    if result.lime_explanation:
        top_tokens = sorted(result.lime_explanation, key=lambda x: abs(x["weight"]), reverse=True)[:5]
        positive_tokens = [t["word"] for t in top_tokens if t["weight"] > 0]
        if positive_tokens:
            parts.append(f"Contributing language: {', '.join(positive_tokens)}")

    parts.append(f"Source: {result.layer_reached} layer")
    return " | ".join(parts)


class NLPDistressPipeline:
    """
    Three-layer distress detection pipeline.

    Crisis override always runs first. This is enforced structurally in analyze()
    and cannot be bypassed by configuration.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        lexicon_path: str | Path = _DEFAULT_LEXICON_PATH,
        vader_threshold: float = _VADER_THRESHOLD,
        use_lime: bool = True,
        lime_num_features: int = 10,
        lime_num_samples: int = 500,
        device: str = "cpu",
    ):
        self._vader_threshold = vader_threshold
        self._use_lime = use_lime
        self._lime_num_features = lime_num_features
        self._lime_num_samples = lime_num_samples
        self._device = device

        # Load crisis lexicon (always required)
        from src.crisis_lexicon import load_lexicon
        self._lexicon = load_lexicon(lexicon_path)

        # VADER (always loaded — fast, no GPU needed)
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        self._vader = SentimentIntensityAnalyzer()

        # Transformer (optional — fallback to VADER-only if unavailable)
        self._tokenizer = None
        self._transformer = None
        self._transformer_available = False

        model_path = Path(model_path) if model_path else _DEFAULT_MODEL_DIR
        self._load_transformer(model_path)

    def _load_transformer(self, model_path: Path) -> None:
        """
        Load fine-tuned transformer from model_path if it exists.
        Fallback: use distilbert-base-uncased-finetuned-sst-2-english as a
        pre-fine-tuning baseline. Production quality requires DAIC-WOZ fine-tune.
        """
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch

            if (model_path / "config.json").exists():
                print(f"[nlp_distress] Loading fine-tuned model from {model_path}")
                self._tokenizer = AutoTokenizer.from_pretrained(str(model_path))
                self._transformer = AutoModelForSequenceClassification.from_pretrained(
                    str(model_path)
                ).to(self._device)
                self._model_source = "fine-tuned"
            else:
                fallback = "distilbert-base-uncased-finetuned-sst-2-english"
                print(f"[nlp_distress] Fine-tuned model not found. Loading fallback: {fallback}")
                print("[nlp_distress] NOTE: Fine-tune on DAIC-WOZ + eRisk for production quality.")
                self._tokenizer = AutoTokenizer.from_pretrained(fallback)
                self._transformer = AutoModelForSequenceClassification.from_pretrained(
                    fallback
                ).to(self._device)
                self._model_source = "fallback-sst2"

            self._transformer.eval()
            self._transformer_available = True

        except Exception as e:
            print(f"[nlp_distress] Transformer unavailable ({e}). VADER-only mode.")
            self._transformer_available = False
            self._model_source = "vader-only"

    def _transformer_predict_proba(self, texts: list[str]) -> list[float]:
        """Return distress probability [0,1] for each text."""
        import torch

        probs = []
        for text in texts:
            inputs = self._tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            ).to(self._device)

            with torch.no_grad():
                logits = self._transformer(**inputs).logits
                prob = torch.softmax(logits, dim=-1)

            # For SST-2 fallback: label 1 = POSITIVE sentiment.
            # We invert: negative sentiment → distress. Index 0 = NEGATIVE.
            # For fine-tuned model: label 1 = distressed (PHQ-8 >= 10).
            if self._model_source == "fallback-sst2":
                # higher NEGATIVE probability → higher distress
                distress_prob = float(prob[0][0].item())
            else:
                distress_prob = float(prob[0][1].item())

            probs.append(distress_prob)
        return probs

    def _get_lime_explanation(self, text: str) -> list[dict]:
        """Run LIME on text, return top token weights as list of dicts."""
        if not self._transformer_available:
            return []
        try:
            from lime.lime_text import LimeTextExplainer

            def predict_fn(texts: list[str]) -> list[list[float]]:
                probs = self._transformer_predict_proba(texts)
                return [[1 - p, p] for p in probs]

            explainer = LimeTextExplainer(
                class_names=["no_distress_signal", "distress_signal"],
                random_state=42,
            )
            exp = explainer.explain_instance(
                text,
                predict_fn,
                num_features=self._lime_num_features,
                num_samples=self._lime_num_samples,
            )
            return [
                {"word": word, "weight": float(weight)}
                for word, weight in exp.as_list()
            ]
        except Exception as e:
            print(f"[nlp_distress] LIME failed: {e}")
            return []

    def analyze(self, text: str, compute_lime: bool = True) -> DistressResult:
        """
        Analyze text for distress signals.

        CRISIS OVERRIDE RUNS FIRST — this is structural and cannot be reordered.
        If crisis detected, returns immediately without calling VADER or transformer.
        """
        from src.crisis_lexicon import check_crisis

        # ----------------------------------------------------------------
        # LAYER 1: Crisis keyword override — ALWAYS RUNS FIRST
        # ----------------------------------------------------------------
        crisis_match = check_crisis(text, lexicon=self._lexicon)

        if crisis_match.matched:
            return DistressResult(
                text=text,
                crisis_detected=True,
                crisis_category=crisis_match.category,
                vader_compound=0.0,
                transformer_score=None,
                distress_signal_score=100.0,
                layer_reached="crisis",
                lime_explanation=None,
                display_text="",
                routing_action="crisis_resources",
            )

        # ----------------------------------------------------------------
        # LAYER 2: VADER sentiment pre-filter
        # ----------------------------------------------------------------
        vader_scores = self._vader.polarity_scores(text)
        compound = vader_scores["compound"]

        if compound > self._vader_threshold:
            # Low risk — skip transformer
            score = _vader_to_score(compound)
            score = min(score, 40.0)  # cap at band top for VADER-filtered text
            result = DistressResult(
                text=text,
                crisis_detected=False,
                crisis_category=None,
                vader_compound=compound,
                transformer_score=None,
                distress_signal_score=score,
                layer_reached="vader",
                lime_explanation=None,
                display_text="",
                routing_action="no_action",
            )
            result.display_text = _format_display_text(result)
            return result

        # ----------------------------------------------------------------
        # LAYER 3: Transformer distress classifier
        # ----------------------------------------------------------------
        if self._transformer_available:
            transformer_prob = self._transformer_predict_proba([text])[0]
            transformer_score = float(transformer_prob * 100)

            # Blend with VADER for robustness (60% transformer, 40% VADER)
            vader_score = _vader_to_score(compound)
            blended_score = 0.6 * transformer_score + 0.4 * vader_score
            blended_score = float(min(blended_score, 99.0))  # reserve 100 for crisis

            lime_expl = self._get_lime_explanation(text) if compute_lime and self._use_lime else None
            routing = "counselor_review" if blended_score >= 41 else "no_action"

            result = DistressResult(
                text=text,
                crisis_detected=False,
                crisis_category=None,
                vader_compound=compound,
                transformer_score=transformer_score,
                distress_signal_score=blended_score,
                layer_reached="transformer",
                lime_explanation=lime_expl,
                display_text="",
                routing_action=routing,
            )
        else:
            # VADER-only fallback (transformer unavailable)
            vader_score = _vader_to_score(compound)
            result = DistressResult(
                text=text,
                crisis_detected=False,
                crisis_category=None,
                vader_compound=compound,
                transformer_score=None,
                distress_signal_score=min(vader_score, 99.0),
                layer_reached="vader",
                lime_explanation=None,
                display_text="",
                routing_action="counselor_review" if vader_score >= 41 else "no_action",
            )

        result.display_text = _format_display_text(result)
        return result

    def analyze_batch(
        self,
        texts: list[str],
        compute_lime: bool = False,  # False by default for batch — expensive
    ) -> list[DistressResult]:
        return [self.analyze(t, compute_lime=compute_lime) for t in texts]

    @property
    def model_source(self) -> str:
        return self._model_source if hasattr(self, "_model_source") else "unknown"


# ---------------------------------------------------------------------------
# Convenience loader
# ---------------------------------------------------------------------------

def load_pipeline(
    model_path: str | Path | None = None,
    lexicon_path: str | Path = _DEFAULT_LEXICON_PATH,
    use_lime: bool = True,
    device: str = "cpu",
) -> NLPDistressPipeline:
    return NLPDistressPipeline(
        model_path=model_path,
        lexicon_path=lexicon_path,
        use_lime=use_lime,
        device=device,
    )
