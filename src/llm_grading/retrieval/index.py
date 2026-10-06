"""Small in-memory dense index; encoder loaded once."""

import numpy as np


class DenseIndex:
    def __init__(self, texts, settings, encoder=None):
        if encoder is None:
            from sentence_transformers import SentenceTransformer

            encoder = SentenceTransformer(
                settings.get(
                    "embedding_model", "sentence-transformers/all-MiniLM-L6-v2"
                ),
                revision=settings.get("embedding_revision"),
                device=settings.get("device", "cpu"),
            )
        self.encoder = encoder
        self.vectors = np.asarray(encoder.encode(texts, normalize_embeddings=True))
        self.settings = settings
        model_config = (
            getattr(getattr(encoder[0], "auto_model", None), "config", None)
            if hasattr(encoder, "__getitem__")
            else None
        )
        self.metadata = {
            "model": settings.get(
                "embedding_model", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            "revision": getattr(model_config, "_commit_hash", None)
            or settings.get("embedding_revision"),
            "max_sequence_length": getattr(encoder, "max_seq_length", None),
        }

    def scores(self, text):
        vector = np.asarray(self.encoder.encode([text], normalize_embeddings=True))[0]
        return self.vectors @ vector
