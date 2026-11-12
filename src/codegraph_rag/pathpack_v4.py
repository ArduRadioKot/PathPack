"""Public v0.4 retrieval API. No patches, labels or scoring imports."""
from dataclasses import dataclass
from .config import V4Config
from .adaptive_chunker import build_chunk_index
from .hybrid_ranker import HybridIndex
from .context_packer import ContextStream, token_count, COUNTER


@dataclass
class RetrievalResult:
    context: str
    hits: list
    diagnostics: dict

    def to_dict(self):
        return {'context':self.context,'selected':[h.to_dict() for h in self.hits],**self.diagnostics}


class PreparedQuery:
    def __init__(self,index,query,config):
        self.config=config;self.index=index
        self.order,self.candidates,self.seeds=index.rank(query,config)
        self.stream=ContextStream(self.order,allow_partial=config.enable_chunking)

    def pack(self,budget,*,instance_id=None,include_candidates=True):
        context,hits,selected=self.stream.pack(budget)
        candidates=[]
        first_blocked=next((cid for cid,info in selected.items() if info['partial']),None)
        if first_blocked is None:
            first_blocked=next((entry.candidate.chunk.id for entry in self.stream.entries if entry.candidate.chunk.id not in selected),None)
        for candidate in (self.candidates if include_candidates else []):
            cid=candidate.chunk.id
            reason=candidate.reason or self.stream.rejections.get(cid)
            if cid in selected:reason='selected_partial' if selected[cid]['partial'] else 'selected'
            elif not reason:reason='budget_exceeded' if cid==first_blocked else 'after_budget_cutoff'
            candidates.append({**candidate.metadata(),'selected':cid in selected,'decision':reason})
        used=token_count(context)
        diagnostics={'instance_id':instance_id,'method':'pathpack-v4','config':self.config.to_dict(),
                     'budget_est':budget,'tokens_est':used,'remaining_tokens':budget-used,'token_utilization':used/budget,
                     'selected_files':len({h.chunk.path for h in hits}),
                     'selected_functions':len({h.chunk.function_id for h in hits if h.chunk.kind!='module'}),
                     'selected_chunks':len(hits),'token_counter':COUNTER,'seeds':self.seeds,'candidates':candidates,
                     'parse_errors':self.index.chunk_index.errors}
        return RetrievalResult(context,hits,diagnostics)


def retrieve_v4(repository,graph,query,*,budget=2048,config=None):
    cfg=config or V4Config()
    index=HybridIndex(build_chunk_index(repository,graph,cfg),graph)
    return PreparedQuery(index,query,cfg).pack(budget)
