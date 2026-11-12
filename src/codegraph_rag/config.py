"""Validated, serializable configuration; no evaluation labels enter this module."""
from dataclasses import asdict, dataclass, field, replace
import json
from pathlib import Path
import math


@dataclass(frozen=True)
class V4Config:
    target_chunk_tokens: int = 128
    max_chunk_tokens: int = 256
    min_chunk_tokens: int = 32
    chunk_overlap_lines: int = 2
    lexical_weight: float = 0.45
    symbol_weight: float = 0.20
    graph_weight: float = 0.25
    diversity_weight: float = 0.10
    redundancy_strength: float = 0.5
    file_diversity_exponent: float = 1.0
    cost_offset: float = 32.0
    cost_exponent: float = 0.5
    hops: int = 2
    seeds: int = 3
    distance_decay: float = 0.8
    edge_weights: dict = field(default_factory=lambda: {'calls':1.0,'imports':0.45,'inherits':0.65,'defines':0.22})
    candidate_limit: int = 512
    min_relevance: float = 0.0
    enable_graph: bool = True
    enable_diversity: bool = True
    enable_chunking: bool = True
    enable_symbol: bool = True
    enable_cost: bool = True
    graph_direction: str = 'both'

    def __post_init__(self):
        if not 1 <= self.min_chunk_tokens <= self.target_chunk_tokens <= self.max_chunk_tokens:
            raise ValueError('Require 1 <= min <= target <= max chunk tokens')
        if self.chunk_overlap_lines < 0 or self.hops < 0 or self.seeds < 1 or self.candidate_limit < 1:
            raise ValueError('Invalid overlap, hops, seeds or candidate_limit')
        for name in ('lexical_weight','symbol_weight','graph_weight','diversity_weight','cost_offset','cost_exponent','min_relevance','file_diversity_exponent'):
            v=getattr(self,name)
            if not math.isfinite(v) or v < 0: raise ValueError(f'Invalid {name}')
        if self.lexical_weight+self.symbol_weight+self.graph_weight <= 0:
            raise ValueError('At least one evidence weight is required')
        for v in [self.distance_decay,self.redundancy_strength,*self.edge_weights.values()]:
            if not math.isfinite(v) or not 0 <= v <= 1: raise ValueError('Graph weights must be in [0,1]')
        if self.graph_direction not in {'both','out','in'}:
            raise ValueError('graph_direction must be both, out or in')

    def to_dict(self): return asdict(self)

    @classmethod
    def load(cls,path): return cls(**json.loads(Path(path).read_text()))

    def ablation(self,name):
        switches={'full':{},'no-graph':{'enable_graph':False},'no-diversity':{'enable_diversity':False},
                  'no-chunking':{'enable_chunking':False},'no-symbol':{'enable_symbol':False},'no-cost':{'enable_cost':False}}
        if name not in switches: raise ValueError(f'Unknown ablation: {name}')
        return replace(self,**switches[name])

ABLATIONS=('full','no-graph','no-diversity','no-chunking','no-symbol','no-cost')


def _load_mapping(path):
    text=Path(path).read_text(encoding='utf-8')
    if str(path).endswith(('.yaml','.yml')):
        import yaml
        data=yaml.safe_load(text)
    else:
        data=json.loads(text)
    if not isinstance(data,dict):
        raise ValueError(f'Configuration must be a mapping: {path}')
    return data


@dataclass(frozen=True)
class V5Config:
    """Two-stage retrieval. Gold labels are not fields of this object.

    Ordering score, with every component in [0, 1]:
        formula = alpha*FileScore + beta*ChunkScore + gamma*GraphScore + delta*ContextScore
        score = formula * (1 - file_penalty) + first_file_bonus
    FileScore is the stage-1 file evidence. ChunkScore is the fragment match,
    renormalized inside the file when hierarchy is on. GraphScore mixes a typed
    path with overlap against lexically relevant neighbors. ContextScore is
    1 - max Jaccard with chunks already placed. The penalty and the first-file
    bonus are the adaptive diversity rule; they are logged separately from the
    four components. This formula is a hypothesis, not a claimed optimum.
    """
    target_chunk_tokens: int = 128
    max_chunk_tokens: int = 256
    min_chunk_tokens: int = 32
    chunk_overlap_lines: int = 2
    alpha_file: float = 0.20
    beta_chunk: float = 0.50
    gamma_graph: float = 0.18
    delta_context: float = 0.12
    file_lexical_weight: float = 0.55
    file_symbol_weight: float = 0.20
    file_graph_weight: float = 0.25
    chunk_symbol_weight: float = 0.20
    local_chunk_blend: float = 0.72
    diversity_weight: float = 0.12
    local_resistance: float = 0.80
    redundancy_strength: float = 0.45
    file_relative_threshold: float = 0.12
    min_files: int = 8
    max_files: int = 24
    per_file_candidates: int = 16
    candidate_limit: int = 384
    cost_offset: float = 32.0
    cost_exponent: float = 0.40
    hops: int = 2
    seeds: int = 4
    distance_decay: float = 0.8
    edge_weights: dict = field(default_factory=lambda: {'calls':1.0,'imports':0.45,'inherits':0.65,'defines':0.22})
    min_relevance: float = 0.0
    max_anchor_lines: int = 6
    enable_hierarchy: bool = True
    enable_graph: bool = True
    enable_diversity: bool = True
    enable_chunking: bool = True
    enable_file_score: bool = True
    enable_symbol: bool = True
    enable_cost: bool = True
    include_signature_anchor: bool = True
    # v0.5.1. Defaults keep the frozen v0.5 path: a hard file cap and a
    # first-chunk bonus equal to diversity_weight. File evidence "function"
    # scores files from whole functions while the context can stay chunked.
    enable_file_filter: bool = True
    file_evidence: str = 'chunk'
    first_file_bonus: float | None = None

    def __post_init__(self):
        if not 1 <= self.min_chunk_tokens <= self.target_chunk_tokens <= self.max_chunk_tokens:
            raise ValueError('Require 1 <= min <= target <= max chunk tokens')
        if self.chunk_overlap_lines < 0 or self.hops < 0 or self.seeds < 1:
            raise ValueError('Invalid overlap, hops or seeds')
        if self.min_files < 1 or self.max_files < 1 or self.min_files > self.max_files:
            raise ValueError('Require 1 <= min_files <= max_files')
        if self.per_file_candidates < 1 or self.candidate_limit < 1 or self.max_anchor_lines < 0:
            raise ValueError('Invalid candidate or anchor limits')
        if not 0 < self.file_relative_threshold <= 1:
            raise ValueError('file_relative_threshold must be in (0, 1]')
        for name in ('alpha_file','beta_chunk','gamma_graph','delta_context','file_lexical_weight',
                     'file_symbol_weight','file_graph_weight','chunk_symbol_weight','local_chunk_blend',
                     'diversity_weight','local_resistance','redundancy_strength','cost_offset','cost_exponent','min_relevance'):
            value=getattr(self,name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'Invalid {name}')
        if self.alpha_file+self.beta_chunk+self.gamma_graph+self.delta_context <= 0:
            raise ValueError('At least one score coefficient is required')
        if self.chunk_symbol_weight > 1 or self.local_chunk_blend > 1 or self.local_resistance > 1 or self.redundancy_strength > 1:
            raise ValueError('Blend, symbol weight, resistance and redundancy must be <= 1')
        if self.file_evidence not in {'chunk','function'}:
            raise ValueError('file_evidence must be chunk or function')
        if self.first_file_bonus is not None and (not math.isfinite(self.first_file_bonus) or self.first_file_bonus < 0):
            raise ValueError('first_file_bonus must be >= 0')
        for value in [self.distance_decay,*self.edge_weights.values()]:
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError('Graph weights must be in [0, 1]')

    def to_dict(self):
        return asdict(self)

    def to_v4(self):
        """Chunker settings only. Ranking flags stay on the v0.5 config."""
        return V4Config(target_chunk_tokens=self.target_chunk_tokens,max_chunk_tokens=self.max_chunk_tokens,
                        min_chunk_tokens=self.min_chunk_tokens,chunk_overlap_lines=self.chunk_overlap_lines,
                        enable_chunking=self.enable_chunking)

    @classmethod
    def load(cls,path):
        return cls(**_load_mapping(path))

    def ablation(self,name):
        switches={'full':{},'no-hierarchy':{'enable_hierarchy':False},'no-graph':{'enable_graph':False},
                  'no-diversity':{'enable_diversity':False},'no-chunking':{'enable_chunking':False},
                  'no-file-score':{'enable_file_score':False}}
        if name not in switches:
            raise ValueError(f'Unknown v0.5 ablation: {name}')
        return replace(self,**switches[name])


V5_ABLATIONS=('full','no-hierarchy','no-graph','no-diversity','no-chunking','no-file-score')
# v0.5.1 granularity grid. "full" is coarse file evidence and fine context.
V51_ABLATIONS=('full','coarse-coarse','fine-fine','global-fine')


@dataclass(frozen=True)
class V6Config:
    """Evidence-first slices. Gold labels are not fields of this object.

    A whole function supplies the parent prior and the graph seeds. The packed
    context is an AST slice of that function, ranked on its own lexical match.
    No file is removed by a top-k cutoff. Diversity reduces repeated text; it
    does not award a bonus to the first chunk of a new file.
    """
    target_chunk_tokens: int = 128
    max_chunk_tokens: int = 256
    min_chunk_tokens: int = 32
    chunk_overlap_lines: int = 2
    lexical_weight: float = 0.50
    symbol_weight: float = 0.15
    parent_weight: float = 0.20
    graph_weight: float = 0.15
    redundancy_strength: float = 0.50
    same_function_penalty: float = 0.25
    cost_offset: float = 32.0
    cost_exponent: float = 0.5
    hops: int = 3
    seeds: int = 3
    distance_decay: float = 0.8
    edge_weights: dict = field(default_factory=lambda: {'calls':1.0,'imports':0.45,'inherits':0.65,'defines':0.22})
    candidate_limit: int = 512
    min_relevance: float = 0.0
    enable_graph: bool = True
    enable_diversity: bool = True
    enable_chunking: bool = True
    enable_parent: bool = True
    enable_symbol: bool = True
    enable_cost: bool = True
    graph_direction: str = 'both'

    def __post_init__(self):
        if not 1 <= self.min_chunk_tokens <= self.target_chunk_tokens <= self.max_chunk_tokens:
            raise ValueError('Require 1 <= min <= target <= max chunk tokens')
        if self.chunk_overlap_lines < 0 or self.hops < 0 or self.seeds < 1 or self.candidate_limit < 1:
            raise ValueError('Invalid overlap, hops, seeds or candidate_limit')
        for name in ('lexical_weight','symbol_weight','parent_weight','graph_weight','redundancy_strength',
                     'same_function_penalty','cost_offset','cost_exponent','min_relevance'):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'Invalid {name}')
        if self.lexical_weight + self.symbol_weight + self.parent_weight + self.graph_weight <= 0:
            raise ValueError('At least one evidence weight is required')
        if self.redundancy_strength > 1 or self.same_function_penalty > 1:
            raise ValueError('Diversity strengths must be <= 1')
        if self.graph_direction not in {'both','out','in'}:
            raise ValueError('graph_direction must be both, out or in')
        for value in [self.distance_decay, *self.edge_weights.values()]:
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError('Graph weights must be in [0, 1]')

    def to_dict(self):
        return asdict(self)

    def to_v4(self):
        return V4Config(target_chunk_tokens=self.target_chunk_tokens, max_chunk_tokens=self.max_chunk_tokens,
                        min_chunk_tokens=self.min_chunk_tokens, chunk_overlap_lines=self.chunk_overlap_lines,
                        enable_chunking=self.enable_chunking)

    @classmethod
    def load(cls, path):
        return cls(**_load_mapping(path))


V6_ABLATIONS = ('full','hops-1','hops-2','callers','callees','no-graph','no-parent','no-chunking','no-diversity')


@dataclass(frozen=True)
class V7Config:
    """File schedule from v0.4. The first unit of each file is chosen locally.

    A function whose slices fit in whole_max_tokens is packed whole. A larger
    function contributes its best slice. Graph depth stays at the v0.4 setting
    passed in by the caller. Gold labels are not fields of this object.
    """
    whole_max_tokens: int = 192
    enable_hierarchy: bool = True
    enable_adaptive: bool = True
    enable_graph: bool = True
    enable_diversity: bool = True
    enable_budget: bool = True
    cap_to_opening: bool = False

    def __post_init__(self):
        if self.whole_max_tokens < 1:
            raise ValueError('whole_max_tokens must be positive')

    def to_dict(self):
        return asdict(self)

    @classmethod
    def load(cls, path):
        return cls(**_load_mapping(path))


V7_ABLATIONS = ('full', 'no-hierarchy', 'no-adaptive', 'no-graph', 'no-budget', 'no-diversity')


def v7_variant(config, name):
    switches = {
        'full': {},
        'no-hierarchy': {'enable_hierarchy': False},
        'no-adaptive': {'enable_adaptive': False},
        'no-graph': {'enable_graph': False},
        'no-budget': {'enable_budget': False},
        'no-diversity': {'enable_diversity': False},
    }
    if name not in switches:
        raise ValueError(f'Unknown v0.7 variant: {name}')
    return replace(config, **switches[name])


@dataclass(frozen=True)
class V8Config:
    """v0.4 file openers stay put. Only the tokens between them are reallocated.

    A later file cannot be pushed out by a larger slice. Graph depth stays at
    the v0.4 setting passed in by the caller. Gold labels are not fields here.
    """
    whole_max_tokens: int = 192
    enable_deepen: bool = True
    enable_adaptive: bool = True
    enable_graph: bool = True
    enable_diversity: bool = True

    def __post_init__(self):
        if self.whole_max_tokens < 1:
            raise ValueError('whole_max_tokens must be positive')

    def to_dict(self):
        return asdict(self)

    @classmethod
    def load(cls, path):
        return cls(**_load_mapping(path))


V8_ABLATIONS = ('full', 'no-deepen', 'no-adaptive', 'no-graph', 'no-diversity')


@dataclass(frozen=True)
class V9Config:
    """v0.4 ranker with a per-file pool cap and a bonus for new function lines.

    per_file_pool 0 means the original uncapped pool. coverage_bonus 0 leaves
    the v0.4 score unchanged. Gold labels are not fields of this object.
    """
    per_file_pool: int = 8
    coverage_bonus: float = 0.25
    name_prior: float = 0.0
    name_df_max: float = 0.01
    name_min_len: int = 8
    name_df_limit: int = 5
    name_inserts: int = 0
    anchor_tokens: int = 1024
    enable_graph: bool = True
    enable_diversity: bool = True

    def __post_init__(self):
        if self.per_file_pool < 0:
            raise ValueError('per_file_pool must be >= 0')
        if self.coverage_bonus < 0:
            raise ValueError('coverage_bonus must be >= 0')
        if self.name_prior < 0:
            raise ValueError('name_prior must be >= 0')
        if not 0 <= self.name_df_max <= 1:
            raise ValueError('name_df_max must be in [0, 1]')
        if self.name_min_len < 1 or self.name_df_limit < 0 or self.name_inserts < 0 or self.anchor_tokens < 0:
            raise ValueError('Invalid name splice settings')

    def to_dict(self):
        return asdict(self)

    @classmethod
    def load(cls, path):
        return cls(**_load_mapping(path))


V9_ABLATIONS = ('full', 'no-coverage', 'cap-4', 'cap-16', 'no-cap', 'plain')


def v9_variant(config, name):
    switches = {
        'full': {},
        'no-coverage': {'coverage_bonus': 0.0},
        'cap-4': {'per_file_pool': 4},
        'cap-16': {'per_file_pool': 16},
        'no-cap': {'per_file_pool': 0},
        'plain': {'per_file_pool': 0, 'coverage_bonus': 0.0, 'name_prior': 0.0, 'name_inserts': 0},
        'no-name': {'name_prior': 0.0},
    }
    if name not in switches:
        raise ValueError(f'Unknown v0.9 variant: {name}')
    return replace(config, **switches[name])


def v8_variant(config, name):
    switches = {
        'full': {},
        'no-deepen': {'enable_deepen': False},
        'no-adaptive': {'enable_adaptive': False},
        'no-graph': {'enable_graph': False},
        'no-diversity': {'enable_diversity': False},
    }
    if name not in switches:
        raise ValueError(f'Unknown v0.8 variant: {name}')
    return replace(config, **switches[name])


def v6_variant(config, name):
    """Structural checks around the frozen v0.6 formula. Not a weight sweep."""
    switches = {
        'full': {},
        'hops-1': {'hops': 1},
        'hops-2': {'hops': 2},
        'callers': {'graph_direction': 'in'},
        'callees': {'graph_direction': 'out'},
        'no-graph': {'enable_graph': False},
        'no-parent': {'enable_parent': False},
        'no-chunking': {'enable_chunking': False},
        'no-diversity': {'enable_diversity': False},
    }
    if name not in switches:
        raise ValueError(f'Unknown v0.6 variant: {name}')
    return replace(config, **switches[name])


@dataclass(frozen=True)
class V10Config:
    """Evidence-set selection. Gold labels are not fields of this object."""
    mode: str = 'joint'
    nested: bool = False
    enable_structural: bool = True
    enable_semantic: bool = True
    bm25_limit: int = 64
    symbol_limit: int = 64
    path_limit: int = 32
    graph_limit: int = 48
    test_limit: int = 8
    graph_hops: int = 1
    alpha: float = 1.0
    beta: float = 0.35
    gamma: float = 0.25
    delta: float = 0.5
    repeat_weight: float = 0.0
    min_neighbor_lexical: float = 0.15
    beam_width: int = 3
    beam_expand: int = 8
    timeout_sec: float = 8.0

    def __post_init__(self):
        if self.mode not in {'joint', 'coverage', 'paths'}:
            raise ValueError('mode must be joint, coverage or paths')
        if min(self.bm25_limit, self.symbol_limit, self.path_limit, self.graph_limit, self.test_limit) < 0:
            raise ValueError('Channel limits must be >= 0')
        if self.graph_hops < 0 or self.beam_width < 1 or self.beam_expand < 1 or self.timeout_sec <= 0:
            raise ValueError('Invalid search limits')

    def to_dict(self):
        return asdict(self)

    @classmethod
    def load(cls, path):
        return cls(**_load_mapping(path))


V10_ABLATIONS = ('joint', 'coverage', 'paths', 'no-structural', 'nested', 'no-semantic')


def v10_variant(config, name):
    switches = {
        'joint': {'mode': 'joint', 'nested': False, 'enable_structural': True},
        'full': {'mode': 'joint', 'nested': False, 'enable_structural': True},
        'coverage': {'mode': 'coverage', 'enable_structural': False},
        'paths': {'mode': 'paths', 'enable_structural': True},
        'no-structural': {'mode': 'joint', 'enable_structural': False, 'beam_width': 1},
        'nested': {'mode': 'joint', 'nested': True},
        'no-semantic': {'mode': 'joint', 'enable_semantic': False},
    }
    if name not in switches:
        raise ValueError(f'Unknown v1.0 variant: {name}')
    return replace(config, **switches[name])


def v51_variant(config, name):
    """One cell of the granularity experiment. Does not change v0.5 ablations."""
    switches={'full':{},
              'coarse-coarse':{'enable_chunking':False},
              'fine-fine':{'file_evidence':'chunk'},
              'global-fine':{'enable_hierarchy':False,'enable_file_score':False,'file_evidence':'chunk'}}
    if name not in switches:
        raise ValueError(f'Unknown v0.5.1 variant: {name}')
    return replace(config,**switches[name])
