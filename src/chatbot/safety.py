"""
Layer 4: Output regex filter — runs on every LLM response before display.

Blocks clinical diagnosis language even if the LLM generates it despite the system prompt.
Defense-in-depth: system prompt is Layer 3; this filter is Layer 4.

Also provides crisis resource text constants used across Phase 4.
"""

import re
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Crisis resource text (authoritative, used by all layers)
# ---------------------------------------------------------------------------

CRISIS_RESOURCE_TEXT = """**If you are in crisis right now, please reach out:**

- **988 Suicide and Crisis Lifeline** — Call or text **988** (24/7, free, confidential)
- **Crisis Text Line** — Text **HOME to 741741** (24/7, free, confidential)
- **UCLA CAPS after-hours** — **(310) 825-0768** (select after-hours option)
- **If in immediate danger** — Call **911** or go to your nearest emergency room

You don't have to face this alone. A real person is available right now."""

CRISIS_RESOURCE_TEXT_PLAIN = (
    "988 Suicide and Crisis Lifeline: Call or text 988 (24/7). "
    "Crisis Text Line: Text HOME to 741741 (24/7). "
    "UCLA CAPS after-hours: (310) 825-0768. "
    "If in immediate danger: call 911."
)

CRISIS_RESOURCE_TEXT_SHORT = "988 (call/text) | Crisis Text Line: HOME to 741741"

CRISIS_FOOTER_MD = (
    "**Crisis resources:** Call or text **988** · Text HOME to **741741** · "
    "UCLA CAPS: **(310) 825-0768** · Emergency: **911**"
)


# ---------------------------------------------------------------------------
# Output filter patterns
# ---------------------------------------------------------------------------

_DIAGNOSIS_PATTERNS = [
    # "you have X disorder / condition"
    r"\byou\s+(have|'ve\s+got|likely\s+have|probably\s+have|may\s+have|might\s+have|seem\s+to\s+have|appear\s+to\s+have)\s+(depression|anxiety\s+disorder|ptsd|bipolar|ocd|schizophrenia|eating\s+disorder|adhd|borderline|bpd|panic\s+disorder|social\s+anxiety|generalized\s+anxiety)",
    # "this sounds like X"
    r"\bthis\s+(sounds?|looks?|seems?|appears?)\s+like\s+(depression|anxiety\s+disorder|ptsd|bipolar|ocd|schizophrenia|eating\s+disorder|adhd|borderline|bpd|panic\s+disorder)",
    # "that's a symptom of X"
    r"\bthat'?s?\s+a\s+symptom\s+of\b",
    # "you are depressed / anxious"
    r"\byou\s+are\s+(depressed|clinically\s+depressed|severely\s+anxious|bipolar|schizophrenic)\b",
    # "you're experiencing depression"
    r"\byou'?re?\s+experiencing\s+(depression|a\s+depressive\s+episode|anxiety\s+disorder|ptsd|bipolar)",
    # "I think you have" / "I believe you have"
    r"\bI\s+(think|believe|suspect)\s+you\s+(have|might\s+have|may\s+have)\s+(depression|anxiety|ptsd|bipolar|ocd)",
    # "diagnose" / "diagnosis"
    r"\b(diagnos\w*)\b",
]

_COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in _DIAGNOSIS_PATTERNS]


@dataclass
class FilterResult:
    original: str
    filtered: str
    was_modified: bool
    triggered_patterns: list[str]


def filter_output(text: str) -> FilterResult:
    """
    Run output filter on LLM response.
    Replaces diagnosis-like language with a safe redirect.
    Returns FilterResult indicating whether modification occurred.
    """
    modified = text
    triggered = []

    for pattern in _COMPILED_PATTERNS:
        if pattern.search(modified):
            triggered.append(pattern.pattern)
            # Replace the matched segment with a safe redirect
            modified = pattern.sub(
                "[I can't assess clinical conditions — please speak with a counselor for a proper evaluation]",
                modified,
            )

    return FilterResult(
        original=text,
        filtered=modified,
        was_modified=len(triggered) > 0,
        triggered_patterns=triggered,
    )


def contains_crisis_output(text: str) -> bool:
    """Check whether LLM output already contains crisis resource information."""
    crisis_indicators = ["988", "741741", "crisis", "lifeline", "emergency"]
    return any(indicator.lower() in text.lower() for indicator in crisis_indicators)
