"""Data model for a statically analyzed Python repository graph."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path


@dataclass(frozen=True)
class Symbol:
    id: str
    path: str
    module: str
    qualname: str
    kind: str
    line_start: int
    line_end: int


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    kind: str


@dataclass
class CodeGraph:
    nodes: dict[str, Symbol] = field(default_factory=dict)
    edges: set[Edge] = field(default_factory=set)
    errors: dict[str, str] = field(default_factory=dict)

    def add_symbol(self, symbol: Symbol) -> None:
        self.nodes[symbol.id] = symbol

    def add_edge(self, source: str, target: str, kind: str) -> None:
        if source in self.nodes and target in self.nodes:
            self.edges.add(Edge(source, target, kind))

    def stats(self) -> dict[str, object]:
        kinds: dict[str, int] = {}
        for symbol in self.nodes.values():
            kinds[symbol.kind] = kinds.get(symbol.kind, 0) + 1
        relations: dict[str, int] = {}
        for edge in self.edges:
            relations[edge.kind] = relations.get(edge.kind, 0) + 1
        return {
            "nodes": len(self.nodes),
            "edges": len(self.edges),
            "symbols": dict(sorted(kinds.items())),
            "relations": dict(sorted(relations.items())),
            "parse_errors": self.errors,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "0.1",
            "nodes": [asdict(node) for node in sorted(self.nodes.values(), key=lambda x: x.id)],
            "edges": [asdict(edge) for edge in sorted(self.edges, key=lambda x: (x.source, x.kind, x.target))],
            "stats": self.stats(),
        }

    def save(self, destination: str | Path) -> None:
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
