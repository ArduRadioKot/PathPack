"""Optional dense baseline. Never silently replaced by BM25."""
from __future__ import annotations

MODEL_ID='sentence-transformers/all-MiniLM-L6-v2'
MODEL_REVISION='c9745ed1d9f207416be6d2e6f8de32d1f16199bf'
# Revision is the sentence-transformers v2.2.2 tag of all-MiniLM-L6-v2.
# Mean pooling, cosine similarity, no fine-tuning, query and code encoded
# by the same checkpoint. Not a Gemma tokenizer.


def probe_embedding():
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        return {'status':'not_run','model':MODEL_ID,'revision':MODEL_REVISION,
                'reason':'sentence-transformers is not installed. No weights were downloaded and no lexical method was substituted under this name.'}
    return {'status':'available','model':MODEL_ID,'revision':MODEL_REVISION,
            'pooling':'mean','similarity':'cosine','normalized':True}
