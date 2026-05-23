"""
RAG Retriever  (Phase 6 — RAG Knowledge Base)
==============================================

ChromaDB-backed retriever over station_docs/ markdown knowledge files.
Uses local sentence-transformers embeddings — no API key required.

To rebuild the index after editing station_docs/:
    python scripts/build_rag_index.py
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

_BASE      = Path(__file__).parent.parent          # experiment/
_DOCS_DIR  = _BASE / "agent" / "station_docs"
_INDEX_DIR = _BASE / "outputs" / "rag_index"

_INSTANCE: Optional["StationRAG"] = None


class StationRAG:
    """ChromaDB-backed RAG retriever over station_docs/ knowledge files."""

    def __init__(
        self,
        docs_dir:  Path = _DOCS_DIR,
        index_dir: Path = _INDEX_DIR,
    ):
        self._docs_dir  = Path(docs_dir)
        self._index_dir = Path(index_dir)
        self._collection = None

    # ------------------------------------------------------------------ #
    # Singleton                                                            #
    # ------------------------------------------------------------------ #

    @classmethod
    def get_instance(cls) -> "StationRAG":
        global _INSTANCE
        if _INSTANCE is None:
            _INSTANCE = cls()
            _INSTANCE._load_or_build()
        return _INSTANCE

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _get_chroma_client(self):
        import chromadb
        self._index_dir.mkdir(parents=True, exist_ok=True)
        return chromadb.PersistentClient(path=str(self._index_dir))

    def _get_embed_fn(self):
        """Local sentence-transformers embedding function (multilingual)."""
        try:
            from chromadb.utils.embedding_functions import (
                SentenceTransformerEmbeddingFunction,
            )
            return SentenceTransformerEmbeddingFunction(
                model_name="paraphrase-multilingual-MiniLM-L12-v2"
            )
        except Exception:
            from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
            return DefaultEmbeddingFunction()

    def _chunk_file(
        self, filepath: Path, chunk_size: int = 200, overlap: int = 30
    ) -> list[dict]:
        text   = filepath.read_text(encoding="utf-8")
        words  = text.split()
        chunks = []
        i = 0
        while i < len(words):
            phrase = " ".join(words[i : i + chunk_size])
            if phrase.strip():
                chunks.append(
                    {
                        "text":     phrase,
                        "source":   filepath.name,
                        "chunk_id": f"{filepath.stem}_c{i}",
                    }
                )
            i += chunk_size - overlap
        return chunks

    # ------------------------------------------------------------------ #
    # Build / Load                                                         #
    # ------------------------------------------------------------------ #

    def build_index(self) -> int:
        """(Re)build the ChromaDB index from all station_docs/ .md files."""
        client   = self._get_chroma_client()
        embed_fn = self._get_embed_fn()

        try:
            client.delete_collection("station_docs")
        except Exception:
            pass

        collection = client.create_collection(
            name="station_docs",
            embedding_function=embed_fn,
            metadata={"hnsw:space": "cosine"},
        )

        all_chunks: list[dict] = []
        for md_file in sorted(self._docs_dir.glob("*.md")):
            all_chunks.extend(self._chunk_file(md_file))

        if not all_chunks:
            self._collection = collection
            return 0

        # Batch add (ChromaDB recommends ≤ 500 per call)
        batch_size = 200
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i : i + batch_size]
            collection.add(
                ids=[c["chunk_id"] for c in batch],
                documents=[c["text"] for c in batch],
                metadatas=[{"source": c["source"]} for c in batch],
            )

        self._collection = collection
        return len(all_chunks)

    def _load_or_build(self) -> None:
        client   = self._get_chroma_client()
        embed_fn = self._get_embed_fn()
        try:
            col = client.get_collection("station_docs", embedding_function=embed_fn)
            if col.count() > 0:
                self._collection = col
                return
        except Exception:
            pass
        self.build_index()

    # ------------------------------------------------------------------ #
    # Query                                                                #
    # ------------------------------------------------------------------ #

    def query(self, text: str, n_results: int = 3) -> List[str]:
        """Return the top-n most relevant passage strings for *text*."""
        if self._collection is None:
            self._load_or_build()
        total = self._collection.count()
        if total == 0:
            return []
        results = self._collection.query(
            query_texts=[text],
            n_results=min(n_results, total),
            include=["documents"],
        )
        raw = results.get("documents", [[]])[0]
        return [d for d in raw if d]
