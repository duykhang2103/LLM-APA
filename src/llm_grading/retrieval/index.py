"""Small in-memory dense, BM25, and hybrid indexes; encoder loaded once."""

import math
import re
from collections import Counter

import numpy as np

CODE_TOKEN = re.compile(
    r"[A-Za-z_]\w*|\d+|==|!=|<=|>=|&&|\|\||<<|>>|\+\+|--|->|[^\s\w]"
)


def code_tokens(text):
    return CODE_TOKEN.findall(text)


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
            if settings.get("max_seq_length"):
                encoder.max_seq_length = settings["max_seq_length"]
        self.encoder = encoder
        self.vectors = np.asarray(encoder.encode(texts, normalize_embeddings=True))
        self.settings = settings
        model_config = (
            getattr(getattr(encoder[0], "auto_model", None), "config", None)
            if hasattr(encoder, "__getitem__")
            else None
        )
        self.metadata = {
            "index": "dense",
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


class BM25Index:
    """Okapi BM25 over C++ identifier/operator tokens; no download needed."""

    def __init__(self, texts, settings):
        self.k1 = float(settings.get("bm25_k1", 1.5))
        self.b = float(settings.get("bm25_b", 0.75))
        self.docs = [Counter(code_tokens(text)) for text in texts]
        self.lengths = np.array([sum(doc.values()) for doc in self.docs], dtype=float)
        self.average_length = float(self.lengths.mean()) if len(self.docs) else 0.0
        frequency = Counter(token for doc in self.docs for token in doc)
        count = len(self.docs)
        self.idf = {
            token: math.log(1 + (count - n + 0.5) / (n + 0.5))
            for token, n in frequency.items()
        }
        self.metadata = {"index": "bm25", "k1": self.k1, "b": self.b}

    def scores(self, text):
        query = Counter(code_tokens(text))
        result = np.zeros(len(self.docs))
        norm = self.k1 * (1 - self.b + self.b * self.lengths / (self.average_length or 1))
        for token, weight in query.items():
            idf = self.idf.get(token)
            if idf is None:
                continue
            tf = np.array([doc.get(token, 0) for doc in self.docs], dtype=float)
            result += weight * idf * tf * (self.k1 + 1) / (tf + norm)
        return result


class HybridIndex:
    """Reciprocal-rank fusion of dense and BM25 rankings."""

    def __init__(self, texts, settings, encoder=None):
        self.parts = [DenseIndex(texts, settings, encoder), BM25Index(texts, settings)]
        self.rrf_k = int(settings.get("rrf_k", 60))
        self.metadata = {
            "index": "hybrid",
            "rrf_k": self.rrf_k,
            "parts": [part.metadata for part in self.parts],
        }

    def scores(self, text):
        fused = np.zeros(len(self.parts[0].vectors))
        for part in self.parts:
            ranks = np.argsort(np.argsort(-part.scores(text), kind="stable"))
            fused += 1.0 / (self.rrf_k + ranks + 1)
        return fused


def build_index(texts, settings, encoder=None):
    kind = settings.get("index", "dense")
    if kind == "dense":
        return DenseIndex(texts, settings, encoder)
    if kind == "bm25":
        return BM25Index(texts, settings)
    if kind == "hybrid":
        return HybridIndex(texts, settings, encoder)
    raise ValueError("retrieval.index must be dense, bm25, or hybrid")
