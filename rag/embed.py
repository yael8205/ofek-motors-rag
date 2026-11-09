"""Turns text into vectors, with a cache on disk so each text is embedded only once.

Two backends, chosen with the EMBED_BACKEND environment variable:
  local  (default) - intfloat/multilingual-e5-small, free, runs on your computer, supports Hebrew
  voyage           - voyage-4-lite, free tokens with a Voyage AI account (needs VOYAGE_API_KEY)
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np

CACHE_DIR = Path(".cache")
MODELS = {"local": "intfloat/multilingual-e5-small", "voyage": "voyage-4-lite"}


class Embedder:
    def __init__(self, backend: str | None = None):
        self.backend = backend or os.environ.get("EMBED_BACKEND", "local")
        self.model = MODELS[self.backend]
        self.cache_path = CACHE_DIR / f"{self.model.replace('/', '_')}.json"
        self.cache = json.loads(self.cache_path.read_text(encoding="utf-8")) if self.cache_path.exists() else {}
        self._client = None

    def embed(self, texts: list[str], kind: str) -> np.ndarray:
        """kind is "query" for questions and "passage" for chunks. Returns one unit-length row per text."""
        keys = [hashlib.sha1(f"{kind}:{t}".encode("utf-8")).hexdigest() for t in texts]
        missing = [(k, t) for k, t in zip(keys, texts) if k not in self.cache]
        if missing:
            vectors = self._embed_new([t for _, t in missing], kind)
            for (k, _), v in zip(missing, vectors):
                self.cache[k] = [round(float(x), 6) for x in v]
            CACHE_DIR.mkdir(exist_ok=True)
            self.cache_path.write_text(json.dumps(self.cache), encoding="utf-8")
        vecs = np.array([self.cache[k] for k in keys], dtype=np.float32)
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

    def _embed_new(self, texts: list[str], kind: str):
        print(f"  embedding {len(texts)} new texts with {self.model} ...")
        if self.backend == "local":
            from sentence_transformers import SentenceTransformer
            if self._client is None:
                self._client = SentenceTransformer(self.model)
            # e5 models expect these prefixes
            return self._client.encode([f"{kind}: {t}" for t in texts], normalize_embeddings=True)
        import voyageai
        if self._client is None:
            self._client = voyageai.Client()  # reads VOYAGE_API_KEY
        out = []
        for i in range(0, len(texts), 64):
            result = self._client.embed(texts[i:i + 64], model=self.model,
                                        input_type="query" if kind == "query" else "document")
            out.extend(result.embeddings)
        return out
