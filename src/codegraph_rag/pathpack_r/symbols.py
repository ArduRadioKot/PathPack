"""Exact identifiers from an issue: snake_case, CamelCase, qualified names, paths, traceback."""
from __future__ import annotations

from dataclasses import dataclass
import re

_SNAKE = re.compile(r'\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b')
_CAMEL = re.compile(r'\b(?:[A-Z][a-z0-9]+){2,}\b|\b[A-Z]{2,}[a-z][A-Za-z0-9]*\b')
_QUALIFIED = re.compile(r'\b[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+\b')
_PATH = re.compile(r'(?:[\w.-]+/)+[\w.-]+\.py\b')
_TRACE = re.compile(r'File "([^"]+)", line (\d+), in ([^\n]+)')
_BACKTICK = re.compile(r'`([^`\n]{1,160})`')


@dataclass(frozen=True)
class SymbolQuery:
    kind: str
    span: str
    source: str


def _compact(text: str) -> str:
    return ''.join(ch.lower() for ch in text if ch.isalnum())


def extract_symbols(query: str) -> list[SymbolQuery]:
    """Only spans that are written in the issue. No repository is read."""
    text = query or ''
    found: list[SymbolQuery] = []
    seen: set[tuple[str, str]] = set()

    def add(kind, span, source):
        span = ' '.join(str(span).split())
        if not span:
            return
        key = (kind, span.lower())
        if key in seen:
            return
        seen.add(key)
        found.append(SymbolQuery(kind, span, source))

    for match in _TRACE.finditer(text):
        add('path', match.group(1), 'traceback')
        name = match.group(3).strip().strip('<>')
        if name and name != '<module>':
            add('symbol', name, 'traceback')
    for match in _PATH.finditer(text):
        add('path', match.group(0), 'path')
    for match in _BACKTICK.finditer(text):
        span = match.group(1).strip()
        if '/' in span and span.endswith('.py'):
            add('path', span, 'backtick')
        elif re.fullmatch(r'[A-Za-z_][A-Za-z0-9_.]*', span):
            add('qualified' if '.' in span else 'symbol', span, 'backtick')
    for match in _QUALIFIED.finditer(text):
        add('qualified', match.group(0), 'qualified')
    for match in _CAMEL.finditer(text):
        add('symbol', match.group(0), 'camel')
    for match in _SNAKE.finditer(text):
        add('symbol', match.group(0), 'snake')
    return found


def path_match(chunk_path: str, span: str) -> bool:
    left = (chunk_path or '').replace('\\', '/').lower()
    right = (span or '').replace('\\', '/').lower()
    if not left or not right or not right.endswith('.py'):
        return False
    return left == right or left.endswith('/' + right) or right.endswith('/' + left)


def symbol_match(qualname: str, path: str, query: SymbolQuery) -> bool:
    """A whole component or a full qualified name. A shared CamelCase piece is not a match."""
    if query.kind == 'path':
        return path_match(path, query.span)
    span = query.span.strip()
    qualname = qualname or ''
    if query.kind == 'qualified' or '.' in span:
        return qualname.lower() == span.lower() or qualname.lower().endswith('.' + span.lower())
    leaf = _compact(span.split('.')[-1])
    if len(leaf) < 4:
        return False
    return any(_compact(part) == leaf for part in qualname.split('.') if part and part != '__module__')


def matches_any(qualname: str, path: str, queries: list[SymbolQuery]) -> bool:
    return any(symbol_match(qualname, path, query) for query in queries)
