"""Fixed PathPack-R configuration. Gold labels are not fields of this object."""
from __future__ import annotations

from dataclasses import asdict, dataclass

from ..config import _load_mapping

# UniXcoder base. Mean-pooled encoder states. CodeT5+ embedding does not
# construct on the installed transformers and is not substituted by noise.
SEMANTIC_MODEL = 'microsoft/unixcoder-base'
# MS MARCO MiniLM is a natural-language passage reranker. It is a zero-shot
# baseline for an issue/code pair, not a source-code reranker.
RERANKER_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
# Public Gemma tokenizers are gated. Recounting stays off until one loads.
GEMMA_TOKENIZER = 'google/gemma-2-2b'


@dataclass(frozen=True)
class PathPackRConfig:
    """Pre-registered objective. Do not fit these values to a measured split."""

    lexical_limit: int = 80
    symbol_limit: int = 256
    semantic_limit: int = 40
    graph_limit: int = 64
    graph_hops: int = 1
    graph_seeds: int = 8
    max_chunks_per_function: int = 8
    semantic_model: str = SEMANTIC_MODEL
    semantic_revision: str = ''
    semantic_max_chars: int = 1200
    semantic_index_cap: int = 4096
    reranker_model: str = RERANKER_MODEL
    reranker_revision: str = ''
    reranker_max_chars: int = 1200
    reranker_query_chars: int = 1500
    reranker_batch: int = 16
    active_rounds: int = 2
    shortlist: int = 6
    max_new_per_round: int = 12
    expand_lexical: float = 0.25
    expand_semantic: float = 0.35
    redirect_factor: float = 0.85
    relative_floor: float = 0.35
    min_relevance: float = 0.0
    max_overlap: float = 0.85
    min_marginal: float = 0.02
    necessity_bonus: float = 0.15
    reject_ratio: float = 0.50
    swap_passes: int = 2
    swap_candidates: int = 8
    fusion_lexical: float = 0.40
    fusion_symbol: float = 0.25
    fusion_semantic: float = 0.20
    fusion_graph: float = 0.15
    gemma_tokenizer: str = GEMMA_TOKENIZER
    gemma_revision: str = ''
    # PathPack-E. Fixed before any validation or holdout measurement.
    edit_model: str = 'output/pathpack_e/model.pkl'
    history_model: str = 'output/pathpack_e/model_history.pkl'
    both_model: str = 'output/pathpack_e/model_both.pkl'
    hier_model: str = 'output/pathpack_e/model_hier.pkl'
    edit_file_bonus: float = 0.25
    edit_min_gain: float = 0.004
    edit_cost_exponent: float = 1.0
    edit_skeleton_weight: float = 0.15

    def __post_init__(self):
        if self.active_rounds < 0 or self.active_rounds > 2:
            raise ValueError('active_rounds must be 0, 1 or 2')
        if min(self.lexical_limit, self.symbol_limit, self.semantic_limit, self.graph_limit) < 0:
            raise ValueError('Channel limits must be >= 0')
        if not 0 <= self.relative_floor <= 1 or not 0 <= self.max_overlap <= 1:
            raise ValueError('Floors and overlap must be in [0, 1]')
        if self.shortlist < 1 or self.max_new_per_round < 1 or self.swap_passes < 0:
            raise ValueError('Invalid search limits')
        if self.semantic_index_cap < 1:
            raise ValueError('semantic_index_cap must be >= 1')

    def to_dict(self):
        return asdict(self)

    @classmethod
    def load(cls, path):
        data = _load_mapping(path)
        known = {field for field in cls.__dataclass_fields__}
        extra = sorted(set(data) - known)
        if extra:
            raise ValueError(f'Unknown PathPack-R config fields: {extra}')
        return cls(**data)
