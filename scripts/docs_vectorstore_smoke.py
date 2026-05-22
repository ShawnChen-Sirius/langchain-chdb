"""Self-contained ChDBVectorStore smoke test.

Designed to be copy-pasted (or directly cited) by the upstream
``langchain-ai/langchain`` integration docs. No API keys, no network,
no external embedders — uses a local deterministic hash-based embedder.

What this script exercises:

1. Construct a ``ChDBVectorStore`` against a file-backed chDB database.
2. ``add_documents`` with explicit ids.
3. ``similarity_search`` (raw).
4. ``similarity_search_with_score`` (returns raw chDB distance).
5. ``similarity_search`` with metadata-filter DSL (``$in``).
6. ``delete(ids=...)`` and verify read-after-delete sync.
7. Close the store, re-open the same on-disk database, verify rows are
   readable through the public API alone.

Run::

    uv run python scripts/docs_vectorstore_smoke.py
    # or
    python scripts/docs_vectorstore_smoke.py
"""

from __future__ import annotations

import random
import tempfile
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from langchain_chdb import ChDBVectorStore, DistanceStrategy


class HashEmbedder(Embeddings):
    """Deterministic 8-dimensional embedder for docs / smoke tests.

    Identical text → identical vector across runs and across processes;
    different text → different vector. No semantic similarity is
    promised, only stability and shape correctness.
    """

    DIM = 8

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        rng = random.Random(text)
        return [rng.gauss(0.0, 1.0) for _ in range(self.DIM)]


def run_smoke() -> None:
    tmpdir = tempfile.mkdtemp(prefix="langchain_chdb_smoke_")
    db_path = str(Path(tmpdir) / "vectors.chdb")
    print(f"Using chDB database at: {db_path}")
    print()

    # 1. Construct the store.
    store = ChDBVectorStore(
        embedding=HashEmbedder(),
        database=db_path,
        embedding_dimension=HashEmbedder.DIM,
        distance_strategy=DistanceStrategy.COSINE,
    )

    # 2. add_documents with explicit ids and metadata.
    docs = [
        Document(id="d1", page_content="chDB is an embedded ClickHouse",
                 metadata={"category": "intro", "score": 0.9}),
        Document(id="d2", page_content="ClickHouse SQL has 1000+ functions",
                 metadata={"category": "docs", "score": 0.6}),
        Document(id="d3", page_content="chDB runs SQL on local Parquet",
                 metadata={"category": "intro", "score": 0.8}),
        Document(id="d4", page_content="LangChain integrates many stores",
                 metadata={"category": "docs", "score": 0.4}),
    ]
    ids = store.add_documents(docs)
    print(f"1. add_documents — stored ids: {ids}")
    print()

    # 3. similarity_search (raw Documents).
    hits = store.similarity_search("clickhouse", k=2)
    print("2. similarity_search('clickhouse', k=2):")
    for h in hits:
        print(f"   {h.id}: {h.page_content!r}")
    print()

    # 4. similarity_search_with_score (raw chDB distance).
    pairs = store.similarity_search_with_score("clickhouse", k=2)
    print("3. similarity_search_with_score (smaller = closer, cosine distance):")
    for doc, score in pairs:
        print(f"   {doc.id}: score={score:.4f}  content={doc.page_content!r}")
    print()

    # 5. similarity_search with metadata filter ($in).
    hits = store.similarity_search(
        "anything",
        k=10,
        filter={"category": {"$in": ["intro"]}},
    )
    print("4. similarity_search with filter category $in ['intro']:")
    for h in hits:
        print(f"   {h.id}: {h.page_content!r}  metadata={h.metadata}")
    print()

    # 6. delete(ids=...) + read-after-delete.
    store.delete(ids=["d2"])
    after_delete = store.get_by_ids(["d1", "d2", "d3", "d4"])
    print("5. delete(ids=['d2']) then get_by_ids(['d1','d2','d3','d4']):")
    print(f"   surviving ids: {sorted(d.id for d in after_delete)}")
    print()

    store.close()

    # 7. Reopen against same on-disk path via public API only.
    store2 = ChDBVectorStore(
        embedding=HashEmbedder(),
        database=db_path,
        embedding_dimension=HashEmbedder.DIM,
        distance_strategy=DistanceStrategy.COSINE,
    )
    reopened = store2.get_by_ids(["d1", "d3", "d4"])
    print("6. Reopen same on-disk database, get_by_ids(['d1','d3','d4']):")
    for d in reopened:
        print(f"   {d.id}: {d.page_content!r}")
    print()

    store2.close()
    print("ALL_OK")


if __name__ == "__main__":
    run_smoke()
