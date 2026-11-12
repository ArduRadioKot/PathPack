"""Gemma tokenizer recount. Missing or gated weights are ``not_run``, never a mixed comparison."""
from __future__ import annotations

from .config import GEMMA_TOKENIZER


def probe_gemma(config=None, *, allow_download=False):
    model_id = GEMMA_TOKENIZER if config is None else config.gemma_tokenizer
    revision = '' if config is None else config.gemma_revision
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return {
            'status': 'not_run', 'model': model_id, 'revision': revision,
            'reason': 'transformers is not installed. Every compared method keeps utf8_bytes_div4_ceil_serialized_context_v1.',
        }
    kwargs = {'local_files_only': not allow_download}
    if revision:
        kwargs['revision'] = revision
    try:
        AutoTokenizer.from_pretrained(model_id, **kwargs)
    except Exception as exc:
        return {
            'status': 'not_run', 'model': model_id, 'revision': revision,
            'reason': f'{type(exc).__name__}: {str(exc).splitlines()[0][:240]}',
            'note': 'No method was recounted with a different counter.',
        }
    return {'status': 'available', 'model': model_id, 'revision': revision or 'loaded'}


def count_texts(config, texts, *, allow_download=False):
    """Count every text with the same tokenizer. Returns ``None`` when it cannot run."""
    probe = probe_gemma(config, allow_download=allow_download)
    if probe['status'] != 'available':
        return None, probe
    from transformers import AutoTokenizer
    kwargs = {'local_files_only': not allow_download}
    if config.gemma_revision:
        kwargs['revision'] = config.gemma_revision
    tokenizer = AutoTokenizer.from_pretrained(config.gemma_tokenizer, **kwargs)
    counts = [len(tokenizer.encode(text or '', add_special_tokens=False)) for text in texts]
    return counts, probe
