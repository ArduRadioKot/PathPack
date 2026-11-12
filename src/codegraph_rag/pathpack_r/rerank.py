"""Zero-shot issue/code reranker. A missing model is ``not_run`` and is not replaced."""
from __future__ import annotations

import math

from .config import RERANKER_MODEL


def probe_reranker():
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        return {
            'status': 'not_run',
            'model': RERANKER_MODEL,
            'code_specialized': False,
            'reason': 'sentence-transformers is not installed. No weights were downloaded and no heuristic score was substituted for this model.',
        }
    return {
        'status': 'library_present',
        'model': RERANKER_MODEL,
        'code_specialized': False,
        'reason': '',
    }


def _unit(scores):
    """Map a logit or a probability into (0, 1). Probabilities already in range stay put."""
    values = [float(score) for score in scores]
    if any(score < 0 or score > 1 for score in values):
        return [1.0 / (1.0 + math.exp(-max(min(score, 20.0), -20.0))) for score in values]
    return values


class Reranker:
    def __init__(self, status, model=None, info=None):
        self.status = status
        self.model = model
        self.info = info or {}
        self.cache = {}

    @property
    def available(self):
        return self.status == 'available' and self.model is not None

    def score(self, query, regions, config):
        """Return one score per region, or ``None`` when the model did not run."""
        if not self.available:
            return None
        missing = []
        order = []
        for region in regions:
            key = region.chunk.id
            order.append(key)
            if key not in self.cache:
                missing.append(region)
        if missing:
            pairs = []
            left = (query or '')[:config.reranker_query_chars]
            for region in missing:
                body = (region.chunk.body or '')[:config.reranker_max_chars]
                right = f'{region.chunk.path}\n{region.chunk.qualname}\n{region.chunk.signature}\n{body}'
                pairs.append((left, right))
            raw = self.model.predict(pairs, batch_size=max(1, config.reranker_batch), show_progress_bar=False)
            for region, score in zip(missing, _unit(raw)):
                self.cache[region.chunk.id] = float(score)
        return [self.cache[key] for key in order]


def load_reranker(config, *, allow_download=False):
    base = probe_reranker()
    if base['status'] == 'not_run':
        return Reranker('not_run', info=base)
    from sentence_transformers import CrossEncoder
    import os
    if not allow_download:
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
    kwargs = {'local_files_only': not allow_download}
    if config.reranker_revision:
        kwargs['revision'] = config.reranker_revision
    try:
        model = CrossEncoder(config.reranker_model, **kwargs)
    except Exception as exc:
        return Reranker('not_run', info={
            **base, 'status': 'not_run',
            'reason': f'{type(exc).__name__}: {str(exc).splitlines()[0][:300]}',
            'download': allow_download,
        })
    loaded = getattr(getattr(model, 'model', None), 'config', None)
    revision = getattr(loaded, '_commit_hash', None) or config.reranker_revision
    if config.reranker_revision and revision and revision != config.reranker_revision:
        raise RuntimeError(f'Reranker revision mismatch: config {config.reranker_revision} loaded {revision}')
    reranker = Reranker('available', model, {
        'status': 'available',
        'model': config.reranker_model,
        'revision': revision or '',
        'code_specialized': False,
        'training_data': 'MS MARCO passages, not source code',
        'note': 'Zero-shot natural-language cross-encoder used on an issue/code pair. Not a code-specialized reranker.',
    })
    probe_pairs = [
        ('empty payload crash in handle_request', 'def handle_request(data):\n    return data\n'),
        ('empty payload crash in handle_request', 'def unrelated_zzzz():\n    return 0\n'),
    ]
    raw = model.predict(probe_pairs, batch_size=2, show_progress_bar=False)
    scored = _unit(raw)
    reranker.info['code_probe'] = {
        'matching_name': scored[0],
        'unrelated': scored[1],
        'matching_name_ranks_higher': scored[0] > scored[1],
        'limitation': 'The probe only checks that the model accepts code text and prefers a named function on one synthetic pair.',
    }
    reranker.cache.clear()
    return reranker
