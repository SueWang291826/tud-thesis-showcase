"""
Node Vector Index  (Phase 6 — Semantic Node Search)
====================================================

FAISS index over all navigation graph nodes, built with local
sentence-transformers embeddings.  Enables natural-language node
lookup in tool_layer._resolve() when exact-match patterns fail.

To rebuild after graph changes:
    python scripts/build_node_index.py
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, List, Optional

_BASE        = Path(__file__).parent.parent            # experiment/
_INDEX_PATH  = _BASE / "outputs" / "node_vector_index.pkl"
_MODEL_NAME  = "paraphrase-multilingual-MiniLM-L12-v2"

_INSTANCE: Optional["NodeVectorIndex"] = None


# ------------------------------------------------------------------ #
# Node text generator                                                  #
# ------------------------------------------------------------------ #

def _node_text(node_id: str, data: dict) -> str:
    """Produce a human-readable description for one graph node."""
    parts: list[str] = []

    level = data.get("level", "")
    if level:
        _lvl = {
            "F1": "platform level 站台层 F1",
            "F3": "concourse level 站厅层 F3",
            "F4": "transport hub 交通层 F4",
            "F2": "equipment level 设备层 F2",
        }
        parts.append(_lvl.get(level, level))

    nt = data.get("node_type", "")
    if nt:
        parts.append(nt.replace("_", " "))

    eg = data.get("entrance_group", "")
    if eg:
        label = eg.replace("entrance_", "").upper()
        parts.append(f"entrance gate {label} 入口{label}")

    sl = data.get("semantic_label", "")
    if sl:
        parts.append(sl.lower().replace("_", " "))

    ct = data.get("connector_type", "")
    if ct:
        parts.append(ct.replace("_", " "))

    if data.get("blind_path"):
        parts.append("tactile path 盲道 accessible wheelchair disabled")

    return " | ".join(parts) if parts else f"node {node_id[:24]}"


# ------------------------------------------------------------------ #
# NodeVectorIndex                                                      #
# ------------------------------------------------------------------ #

class NodeVectorIndex:
    """FAISS-backed semantic index over all navigation graph nodes."""

    def __init__(self) -> None:
        self._index    = None        # faiss.IndexFlatIP
        self._node_ids: list[str] = []
        self._built    = False
        self._model    = None        # cached SentenceTransformer

    # ── Singleton ──────────────────────────────────────────────────── #

    @classmethod
    def get_instance(cls, graph: Any) -> "NodeVectorIndex":
        global _INSTANCE
        if _INSTANCE is None:
            _INSTANCE = cls()
        if not _INSTANCE._built:
            _INSTANCE._load_or_build(graph)
        return _INSTANCE

    # ── Model ──────────────────────────────────────────────────────── #

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(_MODEL_NAME)
        return self._model

    # ── Build ──────────────────────────────────────────────────────── #

    def build(self, graph: Any) -> int:
        """Build FAISS index from graph nodes; persist to disk."""
        import numpy as np
        try:
            import faiss
        except ImportError:
            raise RuntimeError("faiss-cpu is required: pip install faiss-cpu")

        model    = self._get_model()
        node_ids = list(graph.nodes())
        texts    = [_node_text(nid, graph.nodes[nid]) for nid in node_ids]

        # Encode in batches of 512
        all_emb = []
        for i in range(0, len(texts), 512):
            batch = model.encode(
                texts[i : i + 512], normalize_embeddings=True, show_progress_bar=False
            )
            all_emb.append(batch)

        embeddings = np.vstack(all_emb).astype("float32")
        dim        = embeddings.shape[1]

        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        self._index    = index
        self._node_ids = node_ids
        self._built    = True

        # Persist (store embeddings so we can reload without graph)
        _INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_INDEX_PATH, "wb") as f:
            pickle.dump({"node_ids": node_ids, "embeddings": embeddings}, f)

        return len(node_ids)

    def _load_or_build(self, graph: Any) -> None:
        if _INDEX_PATH.exists():
            try:
                import numpy as np, faiss
                with open(_INDEX_PATH, "rb") as f:
                    data = pickle.load(f)
                emb = data["embeddings"].astype("float32")
                idx = faiss.IndexFlatIP(emb.shape[1])
                idx.add(emb)
                self._index    = idx
                self._node_ids = data["node_ids"]
                self._built    = True
                return
            except Exception:
                pass  # fall through to rebuild
        self.build(graph)

    # ── Search ─────────────────────────────────────────────────────── #

    def search(self, query: str, k: int = 3) -> List[str]:
        """Return top-k node IDs semantically closest to *query*."""
        if not self._built or self._index is None:
            return []
        import numpy as np
        q = self._get_model().encode(
            [query], normalize_embeddings=True
        ).astype("float32")
        k = min(k, len(self._node_ids))
        _, indices = self._index.search(q, k)
        return [self._node_ids[i] for i in indices[0] if 0 <= i < len(self._node_ids)]
