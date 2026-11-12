"""Local code embeddings. A missing model is ``not_run``. Vectors are never invented."""
from __future__ import annotations

from pathlib import Path
import hashlib

import numpy as np

from .config import SEMANTIC_MODEL


def probe_semantic():
    try:
        import transformers  # noqa: F401
    except ImportError:
        return {
            'status': 'not_run',
            'model': SEMANTIC_MODEL,
            'reason': 'transformers is not installed. No weights were downloaded and no lexical scores were substituted.',
        }
    return {'status': 'library_present', 'model': SEMANTIC_MODEL, 'reason': ''}


def _normalize(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return matrix / norms


class SemanticIndex:
    def __init__(self, status, chunk_ids=None, matrix=None, info=None):
        self.status = status
        self.chunk_ids = list(chunk_ids or [])
        self.matrix = matrix
        self.info = info or {}
        self._position = {cid: i for i, cid in enumerate(self.chunk_ids)}

    @property
    def available(self):
        return self.status == 'available' and self.matrix is not None

    def scores(self, query_vector):
        """Cosine against the stored code vectors. ``None`` when the model did not run."""
        if not self.available or query_vector is None:
            return None
        query = np.asarray(query_vector, dtype=np.float32)
        query = query / max(float(np.linalg.norm(query)), 1e-12)
        values = self.matrix @ query
        return {cid: float(values[i]) for i, cid in enumerate(self.chunk_ids)}


def embed_text(chunk, limit):
    body = (chunk.body or '')[:limit]
    return f'{chunk.path}\n{chunk.qualname}\n{chunk.signature}\n{body}'


def _cache_file(cache_dir, cache_key, revision):
    if not cache_dir or not cache_key:
        return None
    digest = hashlib.sha256(f'{cache_key}:{revision}'.encode()).hexdigest()[:24]
    return Path(cache_dir) / f'semantic_{digest}.npz'


def encode_texts(tokenizer, model, texts, batch):
    import os
    import torch

    threads = min(8, os.cpu_count() or 1)
    torch.set_num_threads(threads)
    # Opt-in accelerator via CODEGRAPH_DEVICE=mps. Default stays cpu so cached vectors keep their meaning.
    device = os.environ.get('CODEGRAPH_DEVICE', 'cpu')
    if device != 'cpu':
        model.to(device)
    vectors = []
    pooling = 'model_embedding'
    model.eval()
    step = max(1, batch)
    with torch.inference_mode():
        for start in range(0, len(texts), step):
            piece = texts[start:start + batch]
            tokens = tokenizer(piece, padding=True, truncation=True, max_length=256, return_tensors='pt')
            output = model(**{key: value.to(device) for key, value in tokens.items()})
            hidden = output[0] if isinstance(output, tuple) else output
            if hasattr(hidden, 'last_hidden_state'):
                hidden = hidden.last_hidden_state
            array = hidden.detach().cpu().numpy()
            if array.ndim == 3:
                mask = tokens['attention_mask'].detach().cpu().numpy()
                weighted = array * mask[:, :, None]
                array = weighted.sum(axis=1) / np.maximum(mask.sum(axis=1, keepdims=True), 1)
                pooling = 'mean'
            elif array.ndim != 2:
                raise RuntimeError(f'Unexpected embedding shape {array.shape}')
            vectors.append(array.astype(np.float32))
            done = min(len(texts), start + step)
            if len(texts) > 1 and (done == len(texts) or (start // step) % 20 == 0):
                print(f'  semantic encode {done}/{len(texts)}', flush=True)
    return _normalize(np.vstack(vectors)), pooling


_MODEL_CACHE = {}


def _load_encoder(config, allow_download):
    key = (config.semantic_model, config.semantic_revision, bool(allow_download))
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    import os
    from transformers import AutoModel, AutoTokenizer
    if not allow_download:
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
    kwargs = {'trust_remote_code': True, 'local_files_only': not allow_download}
    if config.semantic_revision:
        kwargs['revision'] = config.semantic_revision
    tokenizer = AutoTokenizer.from_pretrained(config.semantic_model, **kwargs)
    model = AutoModel.from_pretrained(config.semantic_model, **kwargs)
    loaded = getattr(getattr(model, 'config', None), '_commit_hash', None) or config.semantic_revision
    _MODEL_CACHE[key] = (tokenizer, model, loaded)
    return _MODEL_CACHE[key]


def load_semantic(chunks, config, *, cache_dir=None, cache_key=None, allow_download=False):
    """Encode compact function text. Download only when the benchmark asks for it."""
    base = probe_semantic()
    if base['status'] == 'not_run':
        return SemanticIndex('not_run', info=base)
    try:
        tokenizer, model, loaded = _load_encoder(config, allow_download)
    except Exception as exc:
        return SemanticIndex('not_run', info={
            'status': 'not_run', 'model': config.semantic_model, 'revision': config.semantic_revision,
            'reason': f'{type(exc).__name__}: {str(exc).splitlines()[0][:300]}',
            'download': allow_download,
        })
    loaded = getattr(getattr(model, 'config', None), '_commit_hash', None) or config.semantic_revision
    if config.semantic_revision and loaded and loaded != config.semantic_revision:
        raise RuntimeError(f'Semantic revision mismatch: config {config.semantic_revision} loaded {loaded}')
    texts = [embed_text(chunk, config.semantic_max_chars) for chunk in chunks]
    fingerprint = hashlib.sha256()
    for chunk, text in zip(chunks, texts):
        fingerprint.update(chunk.id.encode())
        fingerprint.update(text.encode())
    fingerprint = fingerprint.hexdigest()
    cache = _cache_file(cache_dir, cache_key, loaded or 'unpinned')
    matrix = None
    pooling = 'cache'
    if cache and cache.is_file():
        try:
            stored = np.load(cache, allow_pickle=False)
            ids = [str(item) for item in stored['ids'].tolist()]
            if str(stored['fingerprint'][0]) == fingerprint and ids == [c.id for c in chunks]:
                matrix = stored['matrix']
        except (OSError, KeyError, ValueError):
            matrix = None
    if matrix is None:
        matrix, pooling = encode_texts(tokenizer, model, texts, config.reranker_batch)
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                cache, matrix=matrix, ids=np.array([c.id for c in chunks]),
                fingerprint=np.array([fingerprint]),
            )
    info = {
        'status': 'available', 'model': config.semantic_model, 'revision': loaded or '',
        'pooling': pooling, 'chunks': len(chunks), 'embedded': len(chunks), 'code_model': True,
        'note': 'UniXcoder base, mean-pooled. CodeT5+ embedding did not construct on transformers 5.22 and was not replaced with noise.',
        'tokenizer': tokenizer, 'model_obj': model,
    }
    return SemanticIndex('available', [c.id for c in chunks], matrix.astype(np.float32), info)


def select_embed_chunks(index, query, config):
    """Exact symbols first, then lexical rank, then stop. The rest are not embedded."""
    from .symbols import extract_symbols, matches_any

    queries = extract_symbols(query)
    lexical = index.code_bm25.scores(query)
    symbols = [chunk for chunk in index.chunks if matches_any(chunk.qualname, chunk.path, queries)]
    symbols.sort(key=lambda chunk: chunk.id)
    ranked = sorted(index.chunks, key=lambda chunk: (-lexical.get(chunk.id, 0.0), chunk.id))
    chosen = []
    seen = set()
    for chunk in symbols:
        if chunk.id in seen:
            continue
        seen.add(chunk.id)
        chosen.append(chunk)
    limit = max(config.semantic_index_cap, len(chosen))
    for chunk in ranked:
        if len(chosen) >= limit:
            break
        if chunk.id in seen or lexical.get(chunk.id, 0.0) <= 0:
            continue
        seen.add(chunk.id)
        chosen.append(chunk)
    report = {
        'embedded': len(chosen),
        'repository_chunks': len(index.chunks),
        'dropped': len(index.chunks) - len(chosen),
        'exact_symbols_kept': len({chunk.id for chunk in symbols}),
        'reason': 'semantic_index_cap' if len(chosen) < len(index.chunks) else '',
    }
    return chosen, report


def prepare_semantic(index, query, config, *, cache_dir=None, cache_key=None, allow_download=False):
    """Embed the capped subset for this query. Unselected chunks get no vector."""
    chosen, report = select_embed_chunks(index, query, config)
    semantic = load_semantic(
        chosen, config, cache_dir=cache_dir, cache_key=cache_key, allow_download=allow_download,
    )
    semantic.info = {**semantic.info, **report}
    return semantic


def encode_query(semantic, config, query):
    """Reuse the already loaded encoder. Does not download and does not invent a vector."""
    if semantic is None or not semantic.available:
        return None
    tokenizer = semantic.info.get('tokenizer')
    model = semantic.info.get('model_obj')
    if tokenizer is None or model is None:
        return None
    matrix, _pooling = encode_texts(tokenizer, model, [query[:config.semantic_max_chars]], 1)
    return matrix[0]
