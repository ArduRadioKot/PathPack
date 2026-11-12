"""Experimental v1.0 with a quality stop instead of a named-evidence stop.

The candidate pool, channels and original v1.0 module are unchanged. This
variant only replaces the selection objective. A repeated evidence shrinks
and does not halt the search. A candidate with its own lexical, symbol,
graph or repository signal can still be added. Selection stops at
``min_utility``, when nothing else fits the budget, or when the pool is empty.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from time import perf_counter

from .config import V10Config, _load_mapping
from .pathpack_v10 import PreparedQueryV10


@dataclass(frozen=True)
class V10EConfig(V10Config):
    """v1.0 pool plus separate signals and a documented quality floor."""

    lexical_weight: float = 0.50
    symbol_weight: float = 0.40
    graph_weight: float = 0.30
    repo_weight: float = 0.25
    min_utility: float = 0.12

    def __post_init__(self):
        super().__post_init__()
        if self.min_utility < 0:
            raise ValueError('min_utility must be >= 0')
        if min(self.lexical_weight, self.symbol_weight, self.graph_weight, self.repo_weight) < 0:
            raise ValueError('Signal weights must be >= 0')

    @classmethod
    def load(cls, path):
        return cls(**_load_mapping(path))


class PreparedQueryV10E(PreparedQueryV10):
    def _gain(self, item, covered, chosen_files, chosen_ids):
        evidence_gain = 0.0
        new_evidence = False
        for evidence_id in item.evidence_ids:
            evidence = self._by_evidence[evidence_id]
            seen = covered[evidence_id]
            if seen == 0:
                evidence_gain += evidence.confidence
                new_evidence = True
            else:
                evidence_gain += self.config.repeat_weight * evidence.confidence / (1 + seen)
        lexical = float(self.lexical.get(item.chunk.id, 0.0))
        symbol = 1.0 if 'symbol' in item.sources else 0.0
        repo = 1.0 if 'path' in item.sources else 0.0
        graph = 0.0
        if self.config.enable_structural and self.config.mode in {'paths', 'joint'}:
            if self._linked(item.chunk.function_id, chosen_ids):
                graph = 1.0
            elif 'graph' in item.sources:
                graph = 0.35
        independent = (
            self.config.lexical_weight * lexical
            + self.config.symbol_weight * symbol
            + self.config.graph_weight * graph
            + self.config.repo_weight * repo
        )
        component = 1.0 if (
            item.chunk.path not in chosen_files and (new_evidence or independent >= self.config.min_utility)
        ) else 0.0
        redundancy = self._overlap(item, chosen_ids)
        utility = (
            self.config.alpha * evidence_gain
            + independent
            + self.config.gamma * component
            - self.config.delta * redundancy
        )
        item.last_signals = {
            'evidence': evidence_gain,
            'lexical': lexical,
            'symbol': symbol,
            'graph': graph,
            'repository': repo,
            'independent': independent,
            'component': component,
            'redundancy': redundancy,
            'utility': utility,
            'new_evidence': new_evidence,
        }
        return utility, new_evidence

    def _select(self, budget):
        started = perf_counter()
        self._by_evidence = {evidence.evidence_id: evidence for evidence in self.evidences}
        # Greedy under the documented floor. The v1.0 beam only reorders an
        # evidence-limited shortlist, so it is not the stop rule under test.
        chosen = self._select_greedy(budget)
        profile = getattr(self, 'profile', None)
        if profile is None:
            self.profile = profile = {}
        profile['selection_sec'] = profile.get('selection_sec', 0.0) + perf_counter() - started
        return chosen

    def _select_greedy(self, budget):
        remaining = list(self.items)
        chosen = []
        covered = Counter()
        files = set()
        ids = set()
        used = 0
        stop = 'exhausted'
        below = 0
        over_budget = 0
        while remaining:
            best = None
            best_key = None
            saw_above_over_budget = False
            saw_below = False
            for item in remaining:
                cost = item.chunk.tokens_est + 16
                utility, _new = self._gain(item, covered, files, ids)
                fits = used + cost <= budget
                if utility < self.config.min_utility:
                    if fits:
                        saw_below = True
                    continue
                if not fits:
                    saw_above_over_budget = True
                    continue
                key = (utility / max(cost, 1) ** 0.5, utility, -cost, item.chunk.id)
                if best_key is None or key > best_key:
                    best_key = key
                    best = item
            if best is None:
                stop = 'budget' if saw_above_over_budget else 'min_utility'
                below += int(saw_below)
                over_budget += int(saw_above_over_budget)
                break
            chosen.append(best)
            remaining = [item for item in remaining if item.chunk.id != best.chunk.id]
            used += best.chunk.tokens_est + 16
            files.add(best.chunk.path)
            if best.chunk.function_id:
                ids.add(best.chunk.function_id)
            for evidence_id in best.evidence_ids:
                covered[evidence_id] += 1
        self.stop_reason = stop
        self.evidence_all_covered = bool(self.evidences) and all(
            covered[evidence.evidence_id] > 0 for evidence in self.evidences
        )
        self.selection_stats = {
            'stop_reason': stop,
            'evidence_all_covered': self.evidence_all_covered,
            'selected': len(chosen),
            'pool': len(self.items),
            'stopped_on_named_evidence': False,
        }
        return chosen

    def pack(self, budget, *, instance_id=None, include_candidates=True):
        result = super().pack(budget, instance_id=instance_id, include_candidates=include_candidates)
        result.diagnostics['method'] = 'pathpack-v10e'
        result.diagnostics['stop_reason'] = getattr(self, 'stop_reason', '')
        result.diagnostics['evidence_all_covered'] = getattr(self, 'evidence_all_covered', False)
        result.diagnostics['selection_stats'] = getattr(self, 'selection_stats', {})
        result.diagnostics['stop_rule'] = 'min_utility_or_budget_or_exhausted'
        if include_candidates:
            by_id = {item.chunk.id: item for item in self.items}
            for row in result.diagnostics['candidates']:
                item = by_id.get(row.get('id'))
                if item is not None:
                    row['signals'] = dict(getattr(item, 'last_signals', {}))
        return result


def retrieve_v10e(repository, graph, query, *, budget=2048, config=None, v4_config=None):
    from .adaptive_chunker import build_chunk_index
    from .hybrid_ranker import HybridIndex

    cfg = config or V10EConfig()
    index = HybridIndex(build_chunk_index(repository, graph, v4_config), graph)
    return PreparedQueryV10E(index, query, cfg, repository).pack(budget)
