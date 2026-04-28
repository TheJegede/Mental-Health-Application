"""
Crisis lexicon loader and match utility.

Used by both Phase 3 (NLP pipeline) and Phase 4 (chatbot).
Crisis detection ALWAYS runs before any classifier inference — this is a
non-negotiable safety constraint from the ethics charter.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path


_DEFAULT_LEXICON_PATH = Path(__file__).parent.parent / "data" / "crisis_keywords.json"

# Cache compiled patterns by dict identity — same dict object reused across calls
_pattern_cache: dict[int, list[tuple[str, re.Pattern]]] = {}


@dataclass(frozen=True)
class CrisisMatch:
    matched: bool
    category: str | None
    pattern: str | None
    snippet: str | None


def load_lexicon(path: str | Path = _DEFAULT_LEXICON_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _compile_patterns(lexicon: dict) -> list[tuple[str, re.Pattern]]:
    compiled = []
    for category, data in lexicon["categories"].items():
        for pattern in data["patterns"]:
            compiled.append((category, re.compile(pattern, re.IGNORECASE)))
    return compiled


def _get_compiled(lexicon: dict) -> list[tuple[str, re.Pattern]]:
    key = id(lexicon)
    if key not in _pattern_cache:
        _pattern_cache[key] = _compile_patterns(lexicon)
    return _pattern_cache[key]


def check_crisis(
    text: str,
    lexicon: dict | None = None,
    lexicon_path: str | Path = _DEFAULT_LEXICON_PATH,
) -> CrisisMatch:
    """
    Run crisis keyword detection against input text.

    This MUST be called before any classifier inference. If this returns
    matched=True, route immediately to crisis resources — do not proceed
    to classification.

    Returns a CrisisMatch with the first matching category and pattern.
    """
    if lexicon is None:
        lexicon = load_lexicon(lexicon_path)

    for category, pattern in _get_compiled(lexicon):
        m = pattern.search(text)
        if m:
            start = max(0, m.start() - 30)
            end = min(len(text), m.end() + 30)
            return CrisisMatch(
                matched=True,
                category=category,
                pattern=pattern.pattern,
                snippet=text[start:end].strip(),
            )

    return CrisisMatch(matched=False, category=None, pattern=None, snippet=None)


def check_crisis_all_matches(
    text: str,
    lexicon: dict | None = None,
    lexicon_path: str | Path = _DEFAULT_LEXICON_PATH,
) -> list[CrisisMatch]:
    """Return all matching categories (for audit/logging). Prefer check_crisis for routing."""
    if lexicon is None:
        lexicon = load_lexicon(lexicon_path)

    matches = []

    for category, pattern in _get_compiled(lexicon):
        m = pattern.search(text)
        if m:
            start = max(0, m.start() - 30)
            end = min(len(text), m.end() + 30)
            matches.append(
                CrisisMatch(
                    matched=True,
                    category=category,
                    pattern=pattern.pattern,
                    snippet=text[start:end].strip(),
                )
            )

    return matches
