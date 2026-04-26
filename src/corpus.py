"""
Corpus loader for the chatbot knowledge base.

Loads markdown files from data/corpus/, parses frontmatter,
and provides filtered access for Phase 4 RAG pipeline.
"""

import re
from pathlib import Path
from dataclasses import dataclass, field


_DEFAULT_CORPUS_PATH = Path(__file__).parent.parent / "data" / "corpus"

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_KV_RE = re.compile(r"^(\w[\w_-]*):\s*(.+)$", re.MULTILINE)


@dataclass
class CorpusDoc:
    path: Path
    title: str
    category: str
    source_url: str
    license: str
    last_verified: str
    crisis_resource: bool
    content: str
    raw_frontmatter: dict = field(default_factory=dict)


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    fm_block = m.group(1)
    body = text[m.end():]
    kv = {}
    for match in _KV_RE.finditer(fm_block):
        key, val = match.group(1), match.group(2).strip()
        if val.lower() in ("true", "false"):
            val = val.lower() == "true"
        kv[key] = val
    return kv, body


def load_corpus(corpus_path: str | Path = _DEFAULT_CORPUS_PATH) -> list[CorpusDoc]:
    corpus_path = Path(corpus_path)
    docs = []
    for md_file in sorted(corpus_path.rglob("*.md")):
        if md_file.name == "_index.md":
            continue
        text = md_file.read_text(encoding="utf-8")
        fm, body = _parse_frontmatter(text)
        docs.append(
            CorpusDoc(
                path=md_file,
                title=fm.get("title", md_file.stem),
                category=fm.get("category", "uncategorized"),
                source_url=fm.get("source_url", ""),
                license=fm.get("license", ""),
                last_verified=fm.get("last_verified", ""),
                crisis_resource=bool(fm.get("crisis_resource", False)),
                content=body.strip(),
                raw_frontmatter=fm,
            )
        )
    return docs


def get_crisis_docs(corpus_path: str | Path = _DEFAULT_CORPUS_PATH) -> list[CorpusDoc]:
    """Return only docs tagged crisis_resource: true. Used for immediate crisis routing."""
    return [d for d in load_corpus(corpus_path) if d.crisis_resource]
