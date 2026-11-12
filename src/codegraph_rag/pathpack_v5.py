"""PathPack v0.5: file retrieval, then fragment retrieval. No gold imports."""
from __future__ import annotations

import time

from .adaptive_chunker import build_chunk_index
from .config import V5Config
from .context_packer import ContextStream, token_count, COUNTER
from .hierarchical_retriever import HierarchicalIndex
from .pathpack_v4 import RetrievalResult


class PreparedQueryV5:
    def __init__(self,index,query,config):
        self.config=config
        self.index=index
        started=time.perf_counter()
        ranked=index.rank(query,config)
        rank_sec=time.perf_counter()-started
        self.order=ranked.order
        self.candidates=ranked.candidates
        self.seeds=ranked.seeds
        self.files=ranked.files
        self.local_blend=ranked.local_blend
        self.timings=dict(ranked.timings)
        self.timings['rank_sec']=rank_sec
        started=time.perf_counter()
        self.stream=ContextStream(self.order,allow_partial=config.enable_chunking)
        self.timings['pack_stream_sec']=time.perf_counter()-started

    def pack(self,budget,*,instance_id=None,include_candidates=True):
        if budget<=0:
            raise ValueError('budget must be positive')
        started=time.perf_counter()
        context,hits,selected=self.stream.pack(budget)
        prefix_sec=time.perf_counter()-started
        candidates=[]
        first_blocked=next((cid for cid,info in selected.items() if info['partial']),None)
        if first_blocked is None:
            first_blocked=next((entry.candidate.chunk.id for entry in self.stream.entries if entry.candidate.chunk.id not in selected),None)
        if include_candidates:
            for candidate in self.candidates:
                cid=candidate.chunk.id
                reason=candidate.reason or self.stream.rejections.get(cid)
                if cid in selected:
                    reason='selected_partial' if selected[cid]['partial'] else 'selected'
                elif not reason:
                    reason='budget_exceeded' if cid==first_blocked else 'after_budget_cutoff'
                candidates.append({**candidate.metadata(),'selected':cid in selected,'decision':reason})
        used=token_count(context)
        profile={**self.timings,'prefix_sec':prefix_sec,
                 'query_sec':self.timings['rank_sec']+self.timings['pack_stream_sec']+prefix_sec}
        diagnostics={'instance_id':instance_id,'method':'pathpack-v5','config':self.config.to_dict(),
                     'budget_est':budget,'tokens_est':used,'remaining_tokens':budget-used,
                     'token_utilization':used/budget if budget else 0.0,
                     'selected_files':len({hit.chunk.path for hit in hits}),
                     'selected_functions':len({hit.chunk.function_id for hit in hits if hit.chunk.kind!='module'}),
                     'selected_chunks':len(hits),'token_counter':COUNTER,'seeds':self.seeds,
                     'files':self.files,'candidates':candidates,'hierarchy':self.config.enable_hierarchy,
                     'local_blend':self.local_blend,'profile':profile,
                     'parse_errors':self.index.chunk_index.errors}
        return RetrievalResult(context,hits,diagnostics)


def retrieve_v5(repository,graph,query,*,budget=2048,config=None):
    from dataclasses import replace
    cfg=config or V5Config()
    chunks=build_chunk_index(repository,graph,cfg.to_v4())
    file_index=None
    if cfg.file_evidence=='function' and cfg.enable_chunking:
        file_index=build_chunk_index(repository,graph,replace(cfg,enable_chunking=False).to_v4())
    index=HierarchicalIndex(chunks,graph,repository,file_chunk_index=file_index)
    return PreparedQueryV5(index,query,cfg).pack(budget)
