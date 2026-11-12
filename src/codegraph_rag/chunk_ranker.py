"""Stage-2 fragment order. One order is shared by every token budget."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class ChunkCandidate:
    chunk: object
    file_score: float
    chunk_score: float
    graph_score: float
    lexical: float
    symbol: float
    local_lexical: float
    graph_path: float
    graph_neighbor: float
    route: list
    edge_kinds: list
    context_score: float = 1.0
    diversity: float = 0.0
    file_penalty: float = 0.0
    formula_score: float = 0.0
    score: float = 0.0
    utility: float = 0.0
    reason: str = ''
    order: int | None = None
    preliminary: float = 0.0
    extra: dict = field(default_factory=dict)

    def metadata(self):
        return {**self.chunk.metadata(),'file_score':self.file_score,'chunk_score':self.chunk_score,
                'graph_score':self.graph_score,'context_score':self.context_score,
                'lexical_score':self.lexical,'symbol_score':self.symbol,'local_lexical':self.local_lexical,
                'graph_path_score':self.graph_path,'graph_neighbor_score':self.graph_neighbor,
                'diversity_contribution':self.diversity,'file_penalty':self.file_penalty,
                'formula_score':self.formula_score,'score':self.score,'utility':self.utility,
                'graph_path':self.route,'edge_kinds':self.edge_kinds,'reason':self.reason,'order':self.order,
                'bm25_score':self.extra.get('bm25',0.0)}


def formula(candidate,context,cfg):
    return (cfg.alpha_file*candidate.file_score+cfg.beta_chunk*candidate.chunk_score
            +cfg.gamma_graph*candidate.graph_score+cfg.delta_context*context)


def utility_of(score,tokens,cfg):
    if not cfg.enable_cost:
        return score
    return score/((tokens+cfg.cost_offset)**cfg.cost_exponent)


def order_chunks(pool,term_sets,cfg):
    """Greedy adaptive diversity. Disabled diversity is a single static sort."""
    remaining=list(pool)
    similarities=defaultdict(float)
    file_counts=defaultdict(int)
    order=[]
    while remaining:
        best=None
        best_key=None
        snapshot=None
        for candidate in remaining:
            context=1.0-similarities[candidate.chunk.id] if cfg.enable_diversity else 1.0
            taken=file_counts[candidate.chunk.path]
            if cfg.enable_diversity and taken:
                saturation=taken/(taken+1.0)
                penalty=saturation*(1.0-cfg.local_resistance*candidate.chunk_score)
            else:
                penalty=0.0
            if penalty<0:
                penalty=0.0
            if penalty>1:
                penalty=1.0
            base=formula(candidate,context,cfg)
            bonus_weight=cfg.diversity_weight if cfg.first_file_bonus is None else cfg.first_file_bonus
            bonus=bonus_weight*context if cfg.enable_diversity and taken==0 else 0.0
            # Redundant text still attenuates the formula; the file penalty does not replace that.
            if cfg.enable_diversity:
                base=base*(1.0-cfg.redundancy_strength*similarities[candidate.chunk.id])
            score=base*(1.0-penalty)+bonus
            utility=utility_of(score,candidate.chunk.tokens_est,cfg)
            key=(-utility,-score,candidate.chunk.id)
            if best_key is None or key<best_key:
                best_key=key
                best=candidate
                snapshot=(context,penalty,base,bonus,score,utility)
        context,penalty,base,bonus,score,utility=snapshot
        best.context_score=context
        best.file_penalty=penalty
        best.formula_score=base
        best.diversity=bonus
        best.score=score
        best.utility=utility
        best.order=len(order)
        order.append(best)
        remaining.remove(best)
        file_counts[best.chunk.path]+=1
        if cfg.enable_diversity:
            chosen=term_sets.get(best.chunk.id,frozenset())
            for candidate in remaining:
                other=term_sets.get(candidate.chunk.id,frozenset())
                overlap=len(chosen & other)/max(1,len(chosen | other))
                similarities[candidate.chunk.id]=max(similarities[candidate.chunk.id],overlap)
    return order
