"""Deterministic evidence from an issue text. No repository and no reference answer."""
from __future__ import annotations

from dataclasses import dataclass, field
import re

from .retrieval import identifier_terms

_BACKTICK = re.compile(r"`([^`\n]{1,120})`")
_EXCEPTION = re.compile(r"\b([A-Z][A-Za-z0-9]*(?:Error|Exception))\b")
_PATH = re.compile(r"(?:[\w.-]+/)+[\w.-]+\.py\b")
_TEST = re.compile(r"\b(test_[A-Za-z0-9_]+)\b")
_TRACE = re.compile(r'File "([^"]+)", line (\d+), in ([^\n]+)')
_TOKEN = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{3,}\b")
_EXPECTED = re.compile(r"\bexpected\b.{0,80}?\bbut\b.{0,80}", re.IGNORECASE)


@dataclass
class Evidence:
    evidence_id: str
    type: str
    source: str
    query_span: str
    confidence: float
    terms: frozenset[str] = field(default_factory=frozenset)
    matched_symbols: list[str] = field(default_factory=list)
    matched_code_nodes: list[str] = field(default_factory=list)

    def to_dict(self):
        return {
            'evidence_id': self.evidence_id,
            'type': self.type,
            'source': self.source,
            'query_span': self.query_span[:160],
            'confidence': self.confidence,
            'matched_symbols': list(self.matched_symbols),
            'matched_code_nodes': list(self.matched_code_nodes),
        }


def _add(found, seen, kind, span, confidence, source):
    span = ' '.join(span.split())
    if not span:
        return
    key = (kind, span.lower())
    if key in seen:
        return
    seen.add(key)
    terms = identifier_terms(span)
    found.append(Evidence(
        evidence_id=f'{kind}:{len(found)}:{span.lower()[:48]}',
        type=kind, source=source, query_span=span, confidence=confidence,
        terms=terms, matched_symbols=sorted(terms),
    ))


def extract_evidence(query: str) -> list[Evidence]:
    """Pull symbols, paths, exceptions and traceback names that the text actually contains."""
    text = query or ''
    found: list[Evidence] = []
    seen: set[tuple[str, str]] = set()
    for match in _BACKTICK.finditer(text):
        _add(found, seen, 'symbol', match.group(1), 1.0, 'backtick')
    for match in _TRACE.finditer(text):
        _add(found, seen, 'path', match.group(1), 1.0, 'traceback')
        _add(found, seen, 'symbol', match.group(3).strip(), 1.0, 'traceback')
    for match in _PATH.finditer(text):
        _add(found, seen, 'path', match.group(0), 0.9, 'path')
    for match in _EXCEPTION.finditer(text):
        _add(found, seen, 'exception', match.group(1), 0.9, 'exception')
    for match in _TEST.finditer(text):
        _add(found, seen, 'test', match.group(1), 0.8, 'test_name')
    for match in _EXPECTED.finditer(text):
        _add(found, seen, 'behavior', match.group(0), 0.4, 'expected_but')
    for match in _TOKEN.finditer(text):
        token = match.group(0)
        if '_' in token or any(ch.isupper() for ch in token[1:]):
            _add(found, seen, 'symbol', token, 0.6, 'identifier')
    return found
