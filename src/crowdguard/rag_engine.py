from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .config import RAGConfig


@dataclass
class RetrievedChunk:
    source: str
    text: str
    score: float


class RAGIndex:
    """Retrieval engine with FAISS/SentenceTransformer and TF-IDF fallback."""

    def __init__(self, config: RAGConfig | None = None):
        self.config = config or RAGConfig()
        self.chunks: List[Tuple[str, str]] = []
        self.embedding_model = None
        self.index = None
        self.tfidf = None
        self.tfidf_matrix = None
        self.backend = "none"

    def _chunk_text(self, text: str) -> List[str]:
        words = text.split()
        size = max(20, self.config.chunk_size_words)
        overlap = min(self.config.chunk_overlap_words, size // 2)
        chunks = []
        start = 0
        while start < len(words):
            end = min(len(words), start + size)
            chunk = " ".join(words[start:end]).strip()
            if chunk:
                chunks.append(chunk)
            if end == len(words):
                break
            start = end - overlap
        return chunks

    def load_documents(self, directory: Optional[str] = None) -> None:
        root = Path(directory or self.config.knowledge_base_dir)
        if not root.exists():
            raise FileNotFoundError(f"Knowledge base folder not found: {root}")
        self.chunks.clear()
        for path in sorted(root.rglob("*")):
            if path.suffix.lower() not in {".txt", ".md"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for chunk in self._chunk_text(text):
                self.chunks.append((path.name, chunk))
        if not self.chunks:
            raise ValueError(f"No .txt or .md documents found in {root}")

    def build(self) -> None:
        if not self.chunks:
            self.load_documents()
        texts = [text for _, text in self.chunks]

        if self.config.use_faiss:
            try:
                import faiss
                from sentence_transformers import SentenceTransformer

                self.embedding_model = SentenceTransformer(self.config.embedding_model)
                embeddings = self.embedding_model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
                embeddings = np.asarray(embeddings, dtype="float32")
                self.index = faiss.IndexFlatIP(embeddings.shape[1])
                self.index.add(embeddings)
                self.backend = "faiss"
                return
            except Exception:
                # Fall through to TF-IDF fallback.
                pass

        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        self._cosine_similarity = cosine_similarity
        self.tfidf = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.tfidf.fit_transform(texts)
        self.backend = "tfidf"

    def search(self, query: str, top_k: Optional[int] = None) -> List[RetrievedChunk]:
        if self.backend == "none":
            self.build()
        top_k = top_k or self.config.top_k

        if self.backend == "faiss":
            q = self.embedding_model.encode([query], normalize_embeddings=True, show_progress_bar=False)
            q = np.asarray(q, dtype="float32")
            scores, indices = self.index.search(q, top_k)
            results: List[RetrievedChunk] = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0:
                    continue
                source, text = self.chunks[int(idx)]
                results.append(RetrievedChunk(source=source, text=text, score=float(score)))
            return results

        q_vec = self.tfidf.transform([query])
        sims = self._cosine_similarity(q_vec, self.tfidf_matrix).ravel()
        top_indices = sims.argsort()[::-1][:top_k]
        return [RetrievedChunk(source=self.chunks[i][0], text=self.chunks[i][1], score=float(sims[i])) for i in top_indices]
