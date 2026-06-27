from __future__ import annotations

from functools import lru_cache

from fastembed import TextEmbedding

_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384


@lru_cache(maxsize=1)
def _model() -> TextEmbedding:
    return TextEmbedding(model_name=_MODEL_NAME)


def embed_texts(texts: list[str]) -> list[list[float]]:
    # ponytail: lazy-loaded singleton; first call downloads ~130 MB model
    return [v.tolist() for v in _model().embed(texts)]
