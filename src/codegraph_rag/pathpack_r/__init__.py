"""PathPack-R: retrieval, reranking and active localization.

Inference code in this package does not read a reference answer. Training lives in
``train.py`` and is not imported by the retriever.
"""

from .pipeline import VARIANTS, execute

__all__ = ['VARIANTS', 'execute']
