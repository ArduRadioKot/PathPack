"""Stage-1 file scores. No gold labels."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FileEvidence:
    path: str
    lexical: float
    symbol: float
    graph: float
    raw: float
    best_chunk_id: str
    best_chunk_score: float
    best_function_id: str
    n_chunks: int


@dataclass(frozen=True)
class FileDecision:
    path: str
    lexical: float
    symbol: float
    graph: float
    raw: float
    score: float
    selected: bool
    reason: str
    best_chunk_id: str
    best_chunk_score: float
    best_function_id: str
    n_chunks: int

    def to_dict(self):
        return {'path':self.path,'file_lexical':self.lexical,'file_symbol':self.symbol,
                'file_graph':self.graph,'file_raw':self.raw,'file_score_raw':self.score,
                'selected':self.selected,'reason':self.reason,'best_chunk_id':self.best_chunk_id,
                'best_chunk_score':self.best_chunk_score,'best_function_id':self.best_function_id,
                'n_chunks':self.n_chunks}


def rank_files(evidence, seed_paths, cfg):
    """Keep every file above a relative score, plus seed files.

    ``max_files`` is a safety cap. Seed files are kept even when they would
    exceed that cap, because a hard top-k was dropping the only moderate hit
    in an otherwise noisy repository. ``enable_file_score`` does not change
    membership: it only zeros the file term inside the stage-2 formula.
    """
    ranked=sorted(evidence,key=lambda item:(-item.raw,item.path))
    best=ranked[0].raw if ranked else 0.0
    decisions=[]
    if not cfg.enable_file_filter:
        for item in ranked:
            score=(item.raw/best) if best>0 else 0.0
            reason='not_filtered' if item.raw>0 else 'no_evidence'
            decisions.append(FileDecision(item.path,item.lexical,item.symbol,item.graph,item.raw,score,
                                          item.raw>0,reason,item.best_chunk_id,item.best_chunk_score,
                                          item.best_function_id,item.n_chunks))
        return decisions,{item.path for item in decisions if item.selected}
    if not cfg.enable_hierarchy:
        decisions=_flat(ranked,best)
        return decisions,{item.path for item in decisions if item.selected}
    selected=[]
    reasons={}
    positive=[item for item in ranked if item.raw>0]
    threshold=cfg.file_relative_threshold*(best if best>0 else 0.0)
    for item in positive:
        if item.path in seed_paths:
            selected.append(item);reasons[item.path]='selected_seed'
    for item in positive:
        if item.path in reasons:
            continue
        if item.raw+1e-12>=threshold:
            selected.append(item);reasons[item.path]='selected_threshold'
    for item in positive:
        if len(selected)>=cfg.min_files:
            break
        if item.path not in reasons:
            selected.append(item);reasons[item.path]='selected_min_files'
    if len(selected)>cfg.max_files:
        seeds=[item for item in selected if reasons[item.path]=='selected_seed']
        others=sorted((item for item in selected if item not in seeds),key=lambda item:(-item.raw,item.path))
        room=max(0,cfg.max_files-len(seeds))
        kept=seeds+others[:room]
        kept_paths={item.path for item in kept}
        for item in selected:
            if item.path not in kept_paths:
                reasons[item.path]='file_cap'
        selected=kept
    chosen={item.path for item in selected}
    for item in ranked:
        score=(item.raw/best) if best>0 else 0.0
        reason=reasons.get(item.path,'below_threshold' if item.raw>0 else 'no_evidence')
        decisions.append(FileDecision(item.path,item.lexical,item.symbol,item.graph,item.raw,score,
                                      item.path in chosen,reason,item.best_chunk_id,item.best_chunk_score,
                                      item.best_function_id,item.n_chunks))
    return decisions, chosen


def _flat(ranked,best):
    decisions=[]
    for item in ranked:
        score=(item.raw/best) if best>0 else 0.0
        reason='hierarchy_disabled' if item.raw>0 else 'no_evidence'
        decisions.append(FileDecision(item.path,item.lexical,item.symbol,item.graph,item.raw,score,
                                      item.raw>0,reason,item.best_chunk_id,item.best_chunk_score,
                                      item.best_function_id,item.n_chunks))
    return decisions
