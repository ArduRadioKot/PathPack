"""Reproducible, dependency-aware code retrieval (no gold-label access).

Baseline: BM25 over non-overlapping source snippets.
PathPack: BM25 seeds, bounded relation-aware graph search and greedy selection
under the same estimated token budget. It is an experimental heuristic, NOT
an established research result or a production-ready fault locator.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
import heapq
import math
from pathlib import Path
import re

from .graph import CodeGraph
from .parser import scan_repository

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|\d+", re.ASCII)
_CAMEL_RE = re.compile(r"([a-z0-9])([A-Z])")
EDGE_WEIGHTS = {"calls": 1.0, "inherits": 0.65, "imports": 0.45, "defines": 0.22}


def terms(text: str) -> list[str]:
    """Split identifiers and prose into lowercase lexical terms (no learned model)."""
    out = []
    for token in _TOKEN_RE.findall(_CAMEL_RE.sub(r"\1 \2", text)):
        out.extend(part.lower() for part in token.split("_") if part)
    return out


_ACRONYM_RE = re.compile(r"([A-Z]+)([A-Z][a-z])")


def identifier_terms(text: str) -> frozenset[str]:
    """Identifier parts, plus the joined form of a split name.

    `KMeans` and `k_means` both yield `kmeans`. `terms` itself is unchanged,
    so the v0.4 lexical index keeps its original tokenization.
    """
    found: list[str] = []
    for token in _TOKEN_RE.findall(text or ""):
        split = _ACRONYM_RE.sub(r"\1_\2", token)
        split = _CAMEL_RE.sub(r"\1_\2", split)
        parts = [part.lower() for part in split.split("_") if part and not part.isdigit()]
        found.extend(parts)
        if len(parts) > 1:
            found.append("".join(parts))
    return frozenset(found)


def estimated_tokens(text: str) -> int:
    """Coarse character-based proxy, NOT a Gemma tokenizer count."""
    return max(1, math.ceil(len(text.encode("utf-8")) / 4))


@dataclass(frozen=True)
class Chunk:
    id: str
    path: str
    qualname: str
    kind: str
    text: str
    tokens_est: int
    function_id: str | None = None
    line_start: int | None = None
    line_end: int | None = None
    source_spans: tuple[tuple[int, int, int], ...] = ()
    signature: str = ""
    parent_id: str | None = None
    body: str = ""

    def metadata(self) -> dict:
        result = {"id": self.id, "path": self.path, "qualname": self.qualname,
                  "kind": self.kind, "tokens_est": self.tokens_est}
        if self.function_id is not None:
            result.update(function_id=self.function_id, line_start=self.line_start,
                          line_end=self.line_end, source_spans=self.source_spans,
                          signature=self.signature, parent_id=self.parent_id)
        return result


@dataclass
class Hit:
    chunk: Chunk
    score: float
    why: str

    def to_dict(self, *, include_text: bool = True) -> dict:
        d = {**self.chunk.metadata(), "score": round(self.score, 6), "why": self.why}
        if include_text:
            d["text"] = self.chunk.text
        return d


class BM25:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.freqs = [Counter(terms(chunk.text + ' ' + chunk.path + ' ' + chunk.qualname))
                      for chunk in chunks]
        self.lengths = [sum(freq.values()) for freq in self.freqs]
        self.average_length = (sum(self.lengths) / len(self.lengths)) if self.lengths else 1
        self.df = Counter(word for freq in self.freqs for word in freq)

    def scores(self, query: str) -> dict[str, float]:
        query_words = sorted(set(terms(query)))
        n = len(self.chunks)
        scores = {}
        for chunk, freq, length in zip(self.chunks, self.freqs, self.lengths):
            score = 0.0
            for word in query_words:
                if word not in freq:
                    continue
                df = self.df[word]
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                tf = freq[word]
                denom = tf + self.k1 * (1 - self.b + self.b * length / max(self.average_length, 1))
                score += idf * tf * (self.k1 + 1) / denom
            scores[chunk.id] = score
        return scores


def make_chunks(repository: str | Path, graph: CodeGraph) -> list[Chunk]:
    """Extract non-overlapping definitions; avoid duplicate class/module bodies.

    Symbols without functions (pure scripts) are covered by a module chunk.
    For functions within classes, include their own method plus signature only.
    Scope: Python source only; annotations/imports outside chunks may be omitted.
    """
    root = Path(repository).resolve()
    by_file: dict[str, list] = defaultdict(list)
    for node in graph.nodes.values():
        by_file[node.path].append(node)
    chunks: list[Chunk] = []
    for rel, nodes in sorted(by_file.items()):
        # Default evaluation measures fix localization in production source,
        # not retrieval of similarly worded tests. Add test retrieval later as
        # a controlled ablation rather than mixing the two silently.
        parts = Path(rel).parts
        if (any(part in {"tests", "test"} for part in parts)
                or Path(rel).name.startswith("test_")
                or Path(rel).name.endswith("_test.py")):
            continue
        full = (root / rel).resolve()
        if not full.is_relative_to(root) or not full.is_file():
            continue
        lines = full.read_text(encoding="utf-8-sig").splitlines(keepends=True)
        # Only outermost functions plus their class methods; skip nested defs,
        # because the outer function already contains them.
        functions = sorted((n for n in nodes if n.kind in {"function", "method"}),
                           key=lambda n: (n.line_start, -n.line_end, n.id))
        chosen = []
        for n in functions:
            if any(p.line_start <= n.line_start and n.line_end <= p.line_end
                   for p in chosen):
                continue
            chosen.append(n)
        if chosen:
            for n in chosen:
                code = "".join(lines[max(0, n.line_start-1):n.line_end]).strip("\n")
                header = f"# {rel}::{n.qualname}\n"
                txt = header + code
                chunks.append(Chunk(n.id, rel, n.qualname, n.kind, txt, estimated_tokens(txt)))
        else:
            module = next((n for n in nodes if n.kind == "module"), None)
            if module:
                txt = f"# {rel}\n" + "".join(lines)
                chunks.append(Chunk(module.id, rel, "__module__", "module",
                                    txt, estimated_tokens(txt)))
    return chunks


def _neighbors(graph: CodeGraph) -> dict[str, list[tuple[str, str, float]]]:
    adj: dict[str, list[tuple[str, str, float]]] = defaultdict(list)
    for e in graph.edges:
        weight = EDGE_WEIGHTS.get(e.kind, 0)
        if weight:
            adj[e.source].append((e.target, e.kind, weight))
            adj[e.target].append((e.source, e.kind, weight))
    return {k: sorted(v) for k, v in adj.items()}


def _relevant_paths(graph: CodeGraph, seeds: list[str], *, hops: int = 3):
    """Max-product path search over a bounded number of graph edges.

    Paths may traverse non-retrievable module/class nodes; only terminal
    chunk nodes consume context budget. Do not expand exponentially.
    """
    adj = _neighbors(graph)
    scores: dict[str, tuple[float, list[str], list[str]]] = {}
    for seed in seeds:
        states = deque([(seed, [seed], [], 1.0)])
        best_state = {(seed, 0): 1.0}
        while states:
            node, path, kinds, weight = states.popleft()
            if (node not in scores or weight > scores[node][0] or
                (weight == scores[node][0] and (len(path), path) <
                 (len(scores[node][1]), scores[node][1]))):
                scores[node] = (weight, path, kinds)
            if len(path) - 1 >= hops:
                continue
            for nxt, kind, edge_weight in adj.get(node, []):
                if nxt in path:
                    continue
                new_w = weight * edge_weight
                key = (nxt, len(path))
                if new_w <= best_state.get(key, 0):
                    continue
                best_state[key] = new_w
                states.append((nxt, path + [nxt], kinds + [kind], new_w))
    return scores


def select_context(chunks: list[Chunk], graph: CodeGraph, query: str, *,
                   method: str = "pathpack", budget: int = 2048, seeds: int = 3,
                   hops: int = 3, graph_bonus: float = 0.75,
                   redundancy_penalty: float = 0.0,
                   lexical_scores: dict[str, float] | None = None) -> list[Hit]:
    """Return chunks under *estimated* token budget, no label access.

    Paths are directed only for visualization. This prototype traverses calls
    undirected for discovery: reverse callers may explain a bug as well.
    """
    if budget <= 0:
        raise ValueError("budget must be > 0")
    if method not in {"bm25", "pathpack", "graph-expand", "pathpack-v3"}:
        raise ValueError("method must be bm25, graph-expand, pathpack or pathpack-v3")
    if seeds < 1 or hops < 0:
        raise ValueError("seeds >= 1 and hops >= 0 required")
    if not 0 <= graph_bonus <= 5:
        raise ValueError("graph_bonus out of bounds")
    if not 0 <= redundancy_penalty <= 1:
        raise ValueError("redundancy_penalty must be between 0 and 1")
    if not chunks:
        return []
    if method == "pathpack-v3":
        if redundancy_penalty != 0:
            raise ValueError("redundancy_penalty applies only to legacy pathpack; pathpack-v3 uses file diversity")
        return _select_v3(chunks, graph, query, budget=budget, seeds=seeds,
                          hops=hops, graph_bonus=graph_bonus, lexical_scores=lexical_scores)
    bm25 = BM25(chunks).scores(query) if lexical_scores is None else lexical_scores
    ranked = sorted(chunks, key=lambda c: (-bm25[c.id], c.id))
    start = [c.id for c in ranked[:seeds] if bm25[c.id] > 0]
    adjacency = _relevant_paths(graph, start, hops=hops) if start and method != "bm25" else {}
    score_max = max(bm25.values(), default=0)
    result: list[Hit] = []
    remaining = budget
    eligible = {c.id: c for c in chunks}
    if method == "graph-expand":
        # Fixed seeds first, then graph-distance ordering of remaining candidates.
        def priority(c):
            proximity = adjacency.get(c.id, (0, [], []))[0]
            return (0 if c.id in start else 1, -proximity, -bm25[c.id], c.id)
        candidates = sorted(chunks, key=priority)
    elif method == "bm25":
        candidates = ranked
    else:
        # Utility rewards BM25 matches and high confidence graph paths.
        # Greedy utility per estimated token (cost-aware packing), not optimal.
        def key(c):
            norm = bm25[c.id] / score_max if score_max > 0 else 0.0
            bridge = adjacency.get(c.id, (0, [], []))[0] if c.id not in start else 0.0
            raw = norm + graph_bonus * bridge
            return (-raw / math.sqrt(c.tokens_est), -raw, c.id)
        candidates = sorted(chunks, key=key)
    # Preserve the top lexical anchor if it fits. Otherwise the size-adjusted
    # greedy ranking can select many small low-relevance functions and omit
    # the issue's entrypoint entirely.
    anchored: set[str] = set()
    if method == "pathpack":
        for first in ranked:
            if bm25[first.id] <= 0:
                break
            if first.tokens_est <= remaining:
                result.append(Hit(first, bm25[first.id] / max(score_max, 1e-12),
                                  "BM25 anchor seed"))
                anchored.add(first.id)
                remaining -= first.tokens_est
                break
    remaining_candidates = [c for c in candidates
                            if c.id not in anchored and c.tokens_est <= remaining]
    selected_terms: list[set[str]] = [set(terms(h.chunk.text)) for h in result]
    while remaining_candidates:
        if method == "pathpack" and redundancy_penalty > 0:
            def adjusted_key(c):
                norm = bm25[c.id] / score_max if score_max > 0 else 0.0
                confidence = adjacency.get(c.id, (0.0, [], []))[0]
                raw = norm + (graph_bonus * confidence if c.id not in start else 0.0)
                candidate_terms = set(terms(c.text))
                redundancy = max((len(candidate_terms & chosen) / max(1, len(candidate_terms | chosen))
                                  for chosen in selected_terms), default=0.0)
                utility = raw * (1.0 - redundancy_penalty * redundancy)
                return (-(utility / math.sqrt(c.tokens_est)), -utility, c.id)
            chunk = min(remaining_candidates, key=adjusted_key)
            remaining_candidates.remove(chunk)
        else:
            chunk = remaining_candidates.pop(0)
        if method == "bm25":
            if bm25[chunk.id] <= 0:
                continue
            reason = "BM25 lexical match"
            score = bm25[chunk.id]
        elif method == "graph-expand":
            confidence, path, kinds = adjacency.get(chunk.id, (0.0, [], []))
            if bm25[chunk.id] <= 0 and confidence <= 0:
                continue
            score = bm25[chunk.id] + confidence
            reason = "graph neighbor: " + " > ".join(path) if path else "BM25 seed"
        else:
            norm = bm25[chunk.id] / score_max if score_max > 0 else 0.0
            confidence, path, kinds = adjacency.get(chunk.id, (0.0, [], []))
            bonus = graph_bonus * confidence if chunk.id not in start else 0.0
            score = norm + bonus
            if score <= 0:
                continue
            reason = ("path: " + " > ".join(path) + " [" + ", ".join(kinds) + "]") if bonus else "BM25 lexical seed"
        result.append(Hit(chunk, score, reason))
        remaining -= chunk.tokens_est
        if method == "pathpack":
            selected_terms.append(set(terms(chunk.text)))
        remaining_candidates = [c for c in remaining_candidates if c.tokens_est <= remaining]
    return result


def retrieve(repository: str | Path, query: str, *, method: str = "pathpack",
             budget: int = 2048, seeds: int = 3, hops: int = 3,
             graph_bonus: float = 0.75,
             redundancy_penalty: float = 0.0, config=None) -> tuple[CodeGraph, list[Hit]]:
    graph = scan_repository(repository)
    if method == "pathpack-v4":
        from .pathpack_v4 import retrieve_v4
        result = retrieve_v4(repository, graph, query, budget=budget, config=config)
        return graph, result.hits
    if method in {"pathpack-v5", "pathpack-v51"}:
        from .pathpack_v5 import retrieve_v5
        result = retrieve_v5(repository, graph, query, budget=budget, config=config)
        return graph, result.hits
    if method == "pathpack-v6":
        from .pathpack_v6 import retrieve_v6
        result = retrieve_v6(repository, graph, query, budget=budget, config=config)
        return graph, result.hits
    if method == "pathpack-v7":
        from .pathpack_v7 import retrieve_v7
        result = retrieve_v7(repository, graph, query, budget=budget, config=config)
        return graph, result.hits
    if method == "pathpack-v8":
        from .pathpack_v8 import retrieve_v8
        result = retrieve_v8(repository, graph, query, budget=budget, config=config)
        return graph, result.hits
    if method == "pathpack-v9":
        from .pathpack_v9 import retrieve_v9
        result = retrieve_v9(repository, graph, query, budget=budget, config=config)
        return graph, result.hits
    if method == "pathpack-v10":
        from .pathpack_v10 import retrieve_v10
        result = retrieve_v10(repository, graph, query, budget=budget, config=config)
        return graph, result.hits
    chunks = make_chunks(repository, graph)
    hits = select_context(chunks, graph, query, method=method, budget=budget,
                          seeds=seeds, hops=hops, graph_bonus=graph_bonus,
                          redundancy_penalty=redundancy_penalty)
    return graph, hits


# v0.3: a fixed, budget-independent ranking for function-aware retrieval.
# Prefix packing is intentionally conservative: smaller-budget selections are
# subsets of larger-budget selections. That property is NOT guaranteed by the
# legacy methods, which may refill budget using different-sized chunks.
def _select_v3(chunks: list[Chunk], graph: CodeGraph, query: str, *,
               budget: int, seeds: int, hops: int, graph_bonus: float,
               lexical_scores: dict[str, float] | None = None) -> list[Hit]:
    lexical = BM25(chunks).scores(query) if lexical_scores is None else lexical_scores
    ranked = sorted(chunks, key=lambda c: (-lexical[c.id], c.id))
    anchors = [c.id for c in ranked[:seeds] if lexical[c.id] > 0]
    paths = _relevant_paths(graph, anchors, hops=hops) if anchors else {}
    max_lex = max(lexical.values(), default=0) or 1.0
    qterms = set(terms(query))
    evidence = []
    for c in chunks:
        confidence, route, kinds = paths.get(c.id, (0, [], []))
        name_words = set(terms(c.path + ' ' + c.qualname))
        match = len(qterms & name_words) / max(1, len(qterms))
        text_score = lexical[c.id] / max_lex
        # Directly selected seed nodes do not get an extra graph bonus.
        graph_score = confidence if c.id not in anchors else 0.0
        relevance = text_score + 1.2 * match + graph_bonus * graph_score
        if relevance <= 0:
            continue
        # Lower exponent than the old cost heuristic to avoid over-selecting
        # tiny, barely relevant functions. Ranking is independent of budget.
        density = relevance / (max(1, c.tokens_est) ** 0.18)
        evidence.append((c, relevance, density, match, graph_score, route, kinds))

    # Build an order once. Diminishing rewards for repeatedly selecting the
    # same file allow a more diverse set of functions to enter the context.
    order = []
    # Priority queue per file: selecting one item changes only that file's
    # next candidate's priority. O(N log F) instead of O(N^2 log N).
    by_file = defaultdict(list)
    for candidate in evidence:
        by_file[candidate[0].path].append(candidate)
    for items in by_file.values():
        items.sort(key=lambda it: (-it[2], -it[1], it[0].id))
    position = defaultdict(int)
    heap = []
    for path, items in by_file.items():
        first = items[0]
        heapq.heappush(heap, (-first[2], -first[1], first[0].id, path))
    while heap:
        _, _, _, path = heapq.heappop(heap)
        index = position[path]
        order.append(by_file[path][index])
        position[path] += 1
        if position[path] < len(by_file[path]):
            next_item = by_file[path][position[path]]
            priority = next_item[2] * (0.78 ** position[path])
            heapq.heappush(heap, (-priority, -next_item[1], next_item[0].id, path))

    result: list[Hit] = []
    remaining = budget
    for c, relevance, density, match, gscore, route, kinds in order:
        # IMPORTANT: stop, do not skip this item, to guarantee nested selected
        # sets for independent 1024/2048/4096-budget runs.
        if c.tokens_est > remaining:
            break
        detail = f'lexical={lexical[c.id]/max_lex:.3f}; symbol={match:.3f}; graph={gscore:.3f}'
        if route and gscore:
            detail += '; path=' + ' > '.join(route) + ' [' + ','.join(kinds) + ']'
        result.append(Hit(c, relevance, 'v3 ' + detail))
        remaining -= c.tokens_est
    return result
