"""Tests for ``ChDBVectorStore`` (Step 3 of the v0.1 execution plan).

Coverage maps to the twelve acceptance criteria from the plan:

1. Idempotent upsert: same id added twice yields one row (the last write wins).
2. ``add_texts(ids=[id])`` twice with different content updates in place.
3. ``add_texts`` without ids generates a UUID and populates ``Document.id``.
4. Embedding dimension mismatch raises ``ValueError`` before any chDB call.
5. ``delete(ids=[...])`` is read-after-write synchronous.
6. Three distance strategies return non-empty, monotonically-ordered results.
7. ``similarity_search_with_relevance_scores`` returns values in ``[0, 1]``.
8. Metadata filter DSL ``{"category": {"$in": ["a", "b"]}}`` narrows correctly.
9. Unknown filter operator raises ``ValueError`` without reaching chDB.
10. async surface matches sync (modulo unordered ties).
11. File-backed persistence: write → close → reopen → search hits previous docs.
12. LangChain ``VectorStoreIntegrationTests`` conformance is a follow-up file.
"""

from __future__ import annotations

import math
import random
import re

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from langchain_chdb import ChDB, ChDBVectorStore, DistanceStrategy

# ---------------------------------------------------------------------------
# embedder fixtures
# ---------------------------------------------------------------------------


class _DeterministicEmbedder(Embeddings):
    """Hash-based deterministic embedder for tests that don't care about
    semantic closeness — same text gives same vector across runs."""

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        rng = random.Random(text)
        return [rng.gauss(0.0, 1.0) for _ in range(self.dim)]


class _OrderedEmbedder(Embeddings):
    """Embedder where any string ending in a number ``n`` produces the
    vector ``[n, 0, 0, ...]``. Tests using this can predict exact
    nearest-neighbour rankings."""

    _NUM_RE = re.compile(r"-?\d+(?:\.\d+)?$")

    def __init__(self, dim: int = 4) -> None:
        self.dim = dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        m = self._NUM_RE.search(text)
        n = float(m.group()) if m else 0.0
        return [n] + [0.0] * (self.dim - 1)


# ---------------------------------------------------------------------------
# Acceptance #1 / #2 — idempotent upsert
# ---------------------------------------------------------------------------


def test_repeated_id_in_same_batch_collapses_to_one_row():
    store = ChDBVectorStore(_DeterministicEmbedder(8), embedding_dimension=8)
    store.add_documents(
        [
            Document(id="1", page_content="first"),
            Document(id="1", page_content="last"),
        ]
    )
    # Exactly one row is stored, not two with id-equal duplicates.
    docs = store.get_by_ids(["1"])
    assert len(docs) == 1
    assert docs[0].page_content == "last"

    # And the physical table has no shadow duplicate — similarity_search
    # must see only one row, not "first" followed by "last".
    hits = store.similarity_search("anything", k=10)
    assert len(hits) == 1
    assert hits[0].page_content == "last"


def test_repeated_id_across_batches_overwrites():
    store = ChDBVectorStore(_DeterministicEmbedder(8), embedding_dimension=8)
    store.add_texts(["a"], ids=["1"])
    store.add_texts(["b"], ids=["1"])
    docs = store.get_by_ids(["1"])
    assert len(docs) == 1
    assert docs[0].page_content == "b"


# ---------------------------------------------------------------------------
# Acceptance #3 — auto-id assigned and back-populated on Document.id
# ---------------------------------------------------------------------------


def test_missing_id_returns_uuid_without_mutating_input():
    """LangChain's VectorStore contract forbids ``add_documents`` from
    mutating the caller's ``Document`` instances. The generated id list
    is returned, and the stored rows are retrievable by those ids, but
    the input ``Document.id`` stays ``None``.
    """
    store = ChDBVectorStore(_DeterministicEmbedder(8), embedding_dimension=8)
    docs = [Document(page_content="x"), Document(page_content="y")]
    ids = store.add_documents(docs)

    assert len(ids) == 2
    # uuid4 is 36-char hex with dashes
    assert all(len(i) == 36 and i.count("-") == 4 for i in ids)

    # Originals untouched.
    assert all(d.id is None for d in docs), (
        "add_documents must not mutate the caller's Document.id"
    )

    # The stored docs are retrievable by the returned ids.
    retrieved = store.get_by_ids(ids)
    assert {r.id for r in retrieved} == set(ids)


# ---------------------------------------------------------------------------
# Acceptance #4 — dimension contract enforced before chDB sees the vector
# ---------------------------------------------------------------------------


class _BadDimEmbedder(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]  # always dim 3

    def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]


def test_dimension_mismatch_raises_before_chdb_call():
    store = ChDBVectorStore(_BadDimEmbedder(), embedding_dimension=8)
    with pytest.raises(ValueError, match=r"[Dd]imension"):
        store.add_texts(["x"])


# ---------------------------------------------------------------------------
# Acceptance #5 — read-after-delete sync
# ---------------------------------------------------------------------------


def test_delete_is_synchronously_visible():
    store = ChDBVectorStore(_DeterministicEmbedder(8), embedding_dimension=8)
    store.add_texts(["alpha", "beta"], ids=["1", "2"])
    assert {d.id for d in store.get_by_ids(["1", "2"])} == {"1", "2"}
    store.delete(ids=["1"])
    remaining = store.get_by_ids(["1", "2"])
    assert {d.id for d in remaining} == {"2"}


def test_delete_with_no_ids_returns_none():
    store = ChDBVectorStore(_DeterministicEmbedder(8), embedding_dimension=8)
    assert store.delete() is None


# ---------------------------------------------------------------------------
# Acceptance #6 — all three distance strategies functional
# ---------------------------------------------------------------------------


class _MapEmbedder(Embeddings):
    """Embedder that returns a fixed vector per text via a lookup map.

    Used for distance-strategy tests so the geometry is fully under test
    control (cosine, L2, and inner-product can rank vectors differently
    when norms vary, so the test layout uses unit vectors at the four
    cardinal directions in 2D and a query identical to one of them).
    """

    def __init__(self, mapping: dict[str, list[float]]) -> None:
        self._mapping = mapping

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._mapping[t] for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._mapping[text]


@pytest.mark.parametrize(
    "strategy",
    [
        DistanceStrategy.COSINE,
        DistanceStrategy.EUCLIDEAN,
        DistanceStrategy.MAX_INNER_PRODUCT,
    ],
)
def test_distance_strategy_returns_ranked_results(strategy):
    """An exact-vector query must surface the matching doc as the top
    hit under every distance strategy.

    Layout: four unit vectors at the cardinal directions, plus a query
    identical to ``doc_east``. Symmetric so cosine, L2, and inner-product
    all agree on the ranking.
    """
    mapping = {
        "doc_east":  [1.0,  0.0],
        "doc_north": [0.0,  1.0],
        "doc_west":  [-1.0, 0.0],
        "doc_south": [0.0, -1.0],
        "query_east": [1.0, 0.0],
    }
    store = ChDBVectorStore(
        _MapEmbedder(mapping),
        embedding_dimension=2,
        distance_strategy=strategy,
    )
    store.add_texts(["doc_east", "doc_north", "doc_west", "doc_south"])
    hits = store.similarity_search("query_east", k=4)
    assert len(hits) == 4
    assert hits[0].page_content == "doc_east"


# ---------------------------------------------------------------------------
# Acceptance #7 — relevance scores fall in [0, 1]
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "strategy",
    [
        DistanceStrategy.COSINE,
        DistanceStrategy.EUCLIDEAN,
        DistanceStrategy.MAX_INNER_PRODUCT,
    ],
)
def test_relevance_scores_in_unit_interval(strategy):
    store = ChDBVectorStore(
        _DeterministicEmbedder(8),
        embedding_dimension=8,
        distance_strategy=strategy,
    )
    store.add_texts(["alpha", "beta", "gamma", "delta", "epsilon"])
    pairs = store.similarity_search_with_relevance_scores("alpha", k=5)
    assert len(pairs) == 5
    for _doc, score in pairs:
        assert 0.0 <= score <= 1.0, f"{strategy.value} produced score {score}"


def test_relevance_score_monotone_with_raw_score():
    """The raw → relevance map must preserve closeness ordering."""
    store = ChDBVectorStore(
        _OrderedEmbedder(4),
        embedding_dimension=4,
        distance_strategy=DistanceStrategy.EUCLIDEAN,
    )
    store.add_texts(["v 1", "v 5", "v 10", "v 100"])
    pairs = store.similarity_search_with_relevance_scores("q 5", k=4)
    relevance = [s for _, s in pairs]
    # Closest first, so relevance must be non-increasing.
    assert relevance == sorted(relevance, reverse=True)


# ---------------------------------------------------------------------------
# Acceptance #8 / #9 — metadata filter DSL
# ---------------------------------------------------------------------------


def _store_with_categorized_docs(strategy=DistanceStrategy.COSINE):
    store = ChDBVectorStore(
        _DeterministicEmbedder(8),
        embedding_dimension=8,
        distance_strategy=strategy,
    )
    store.add_documents(
        [
            Document(id="a", page_content="alpha", metadata={"category": "news", "score": 0.9}),
            Document(id="b", page_content="beta",  metadata={"category": "blog", "score": 0.4}),
            Document(id="c", page_content="gamma", metadata={"category": "news", "score": 0.6}),
            Document(id="d", page_content="delta", metadata={"category": "wiki", "score": 0.7}),
        ]
    )
    return store


def test_filter_equality():
    store = _store_with_categorized_docs()
    hits = store.similarity_search("q", k=10, filter={"category": "news"})
    assert {h.id for h in hits} == {"a", "c"}


def test_filter_in():
    store = _store_with_categorized_docs()
    hits = store.similarity_search("q", k=10, filter={"category": {"$in": ["news", "wiki"]}})
    assert {h.id for h in hits} == {"a", "c", "d"}


def test_filter_comparison():
    store = _store_with_categorized_docs()
    hits = store.similarity_search("q", k=10, filter={"score": {"$gt": 0.5}})
    assert {h.id for h in hits} == {"a", "c", "d"}


def test_filter_ne():
    store = _store_with_categorized_docs()
    hits = store.similarity_search("q", k=10, filter={"category": {"$ne": "blog"}})
    assert {h.id for h in hits} == {"a", "c", "d"}


def test_filter_and():
    store = _store_with_categorized_docs()
    hits = store.similarity_search(
        "q",
        k=10,
        filter={"$and": [{"category": "news"}, {"score": {"$gt": 0.7}}]},
    )
    assert {h.id for h in hits} == {"a"}


def test_filter_or():
    store = _store_with_categorized_docs()
    hits = store.similarity_search(
        "q",
        k=10,
        filter={"$or": [{"category": "blog"}, {"score": {"$gte": 0.9}}]},
    )
    assert {h.id for h in hits} == {"a", "b"}


def test_filter_not():
    store = _store_with_categorized_docs()
    hits = store.similarity_search(
        "q",
        k=10,
        filter={"$not": {"category": "blog"}},
    )
    assert {h.id for h in hits} == {"a", "c", "d"}


@pytest.mark.parametrize(
    "bad_filter, match",
    [
        ({"category": {"$weird": 1}}, r"\$weird"),
        ({"$bogus": [{"x": 1}]}, r"\$bogus"),
        ({"x": {"$in": []}}, r"\$in.*non-empty"),
        ({"x": {"$in": "not-a-list"}}, r"\$in.*list"),
    ],
)
def test_filter_unknown_operators_rejected(bad_filter, match):
    store = ChDBVectorStore(_DeterministicEmbedder(8), embedding_dimension=8)
    store.add_texts(["x"])
    with pytest.raises(ValueError, match=match):
        store.similarity_search("q", k=3, filter=bad_filter)


def test_filter_rejects_invalid_metadata_key():
    """JSON path identifiers must match [A-Za-z_][A-Za-z0-9_]*."""
    store = ChDBVectorStore(_DeterministicEmbedder(8), embedding_dimension=8)
    store.add_texts(["x"])
    with pytest.raises(ValueError, match=r"metadata key"):
        store.similarity_search("q", k=3, filter={"with-dash": "value"})


# ---------------------------------------------------------------------------
# Acceptance #10 — async parity
# ---------------------------------------------------------------------------


async def test_async_add_get_delete_match_sync():
    store = ChDBVectorStore(_DeterministicEmbedder(8), embedding_dimension=8)
    ids = await store.aadd_texts(["a", "b", "c"], ids=["1", "2", "3"])
    assert ids == ["1", "2", "3"]
    docs = await store.aget_by_ids(["1", "2"])
    assert {d.id for d in docs} == {"1", "2"}
    await store.adelete(ids=["2"])
    docs_after = await store.aget_by_ids(["1", "2", "3"])
    assert {d.id for d in docs_after} == {"1", "3"}


async def test_async_search_matches_sync():
    store = ChDBVectorStore(_OrderedEmbedder(4), embedding_dimension=4)
    store.add_texts(["v 1", "v 5", "v 10"])
    sync_hits = store.similarity_search("q 5", k=3)
    async_hits = await store.asimilarity_search("q 5", k=3)
    assert [d.page_content for d in sync_hits] == [d.page_content for d in async_hits]


async def test_async_with_relevance_scores_in_unit_interval():
    store = ChDBVectorStore(_DeterministicEmbedder(8), embedding_dimension=8)
    await store.aadd_texts(["a", "b", "c"])
    pairs = await store.asimilarity_search_with_relevance_scores("a", k=3)
    assert all(0.0 <= s <= 1.0 for _, s in pairs)


# ---------------------------------------------------------------------------
# Acceptance #11 — file-backed persistence round-trip
# ---------------------------------------------------------------------------
#
# chDB initializes one process-global ``EmbeddedServer`` on the first
# ``Session()`` call and refuses to bind a different path afterwards.
# Other tests in this module use ``:memory:``, so by the time we reach
# the persistence test the in-memory path is already locked in. We run
# the round-trip inside a subprocess to get a clean chDB init.


def test_persistence_round_trip(tmp_path):
    """Write to a file-backed store, close, reopen in the same process,
    read the rows back. Runs in a subprocess to dodge the process-global
    ``EmbeddedServer`` initialization other tests in this module trigger."""
    import subprocess
    import sys
    import textwrap

    db = tmp_path / "vec.chdb"
    code = textwrap.dedent(f"""
        from langchain_core.embeddings import Embeddings
        from langchain_chdb import ChDBVectorStore

        class _E(Embeddings):
            def embed_documents(self, texts):
                return [[float(i + 1), 0.0, 0.0, 0.0] for i in range(len(texts))]
            def embed_query(self, text):
                return [1.0, 0.0, 0.0, 0.0]

        # Phase 1 — create the file-backed store, write two rows, close.
        s1 = ChDBVectorStore(_E(), database={str(db)!r}, embedding_dimension=4)
        s1.add_texts(["alpha", "beta"], ids=["1", "2"])
        s1.close()

        # Phase 2 — fresh ChDBVectorStore against the SAME on-disk path.
        # Uses only the public API; the read path must probe the on-disk
        # table without help from private setup.
        s2 = ChDBVectorStore(_E(), database={str(db)!r}, embedding_dimension=4)
        docs = s2.get_by_ids(["1", "2"])
        ids = sorted(d.id for d in docs)
        assert ids == ["1", "2"], f"expected ['1', '2'], got {{ids!r}}"

        # similarity_search must also work without any prior write on s2.
        hits = s2.similarity_search("anything", k=5)
        contents = sorted(h.page_content for h in hits)
        assert contents == ["alpha", "beta"], f"got {{contents!r}}"

        s2.close()
        print("PERSIST_OK")
    """)
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"subprocess failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "PERSIST_OK" in result.stdout


# ---------------------------------------------------------------------------
# from_texts / from_documents constructors
# ---------------------------------------------------------------------------


def test_from_texts_classmethod():
    store = ChDBVectorStore.from_texts(
        ["a", "b"],
        _DeterministicEmbedder(8),
        embedding_dimension=8,
    )
    hits = store.similarity_search("a", k=2)
    assert {d.page_content for d in hits} == {"a", "b"}


def test_from_documents_classmethod_preserves_ids():
    store = ChDBVectorStore.from_documents(
        [Document(id="x", page_content="hello"), Document(id="y", page_content="world")],
        _DeterministicEmbedder(8),
        embedding_dimension=8,
    )
    docs = store.get_by_ids(["x", "y"])
    assert {d.id for d in docs} == {"x", "y"}


# ---------------------------------------------------------------------------
# misc surface checks
# ---------------------------------------------------------------------------


def test_chdb_short_alias_is_vectorstore():
    """``ChDB`` is exported as a short alias for ``ChDBVectorStore``."""
    assert ChDB is ChDBVectorStore


def test_readonly_blocks_writes():
    store = ChDBVectorStore(_DeterministicEmbedder(8), embedding_dimension=8)
    store.add_texts(["x"], ids=["1"])
    store._readonly = True
    with pytest.raises(RuntimeError, match=r"readonly"):
        store.add_texts(["y"])
    with pytest.raises(RuntimeError, match=r"readonly"):
        store.delete(ids=["1"])


def test_get_by_ids_empty_input_short_circuits():
    store = ChDBVectorStore(_DeterministicEmbedder(8), embedding_dimension=8)
    assert store.get_by_ids([]) == []


def test_similarity_search_on_empty_store_returns_empty():
    store = ChDBVectorStore(_DeterministicEmbedder(8), embedding_dimension=8)
    assert store.similarity_search("anything", k=5) == []


def test_relevance_mapping_cosine_clamped_to_zero():
    """Cosine distance can be > 1 for high-dim embeddings; the relevance
    mapping must clamp to ``[0, 1]``."""
    store = ChDBVectorStore(_DeterministicEmbedder(8), embedding_dimension=8)
    # Manually invoke the private mapping on a distance > 1.
    assert store._raw_score_to_relevance(1.5) == 0.0


def test_relevance_mapping_euclidean_is_inverse():
    store = ChDBVectorStore(
        _DeterministicEmbedder(8),
        embedding_dimension=8,
        distance_strategy=DistanceStrategy.EUCLIDEAN,
    )
    assert store._raw_score_to_relevance(0.0) == 1.0
    assert math.isclose(store._raw_score_to_relevance(1.0), 0.5)


def test_score_threshold_filters_low_relevance_pairs():
    """``score_threshold`` drops pairs below the threshold relevance."""
    store = ChDBVectorStore(_DeterministicEmbedder(8), embedding_dimension=8)
    store.add_texts(["alpha", "beta", "gamma", "delta", "epsilon"])

    # Without threshold: all five returned.
    all_pairs = store.similarity_search_with_relevance_scores("alpha", k=5)
    assert len(all_pairs) == 5

    # With a threshold that should exclude at least the tail.
    relevances = [s for _, s in all_pairs]
    cutoff = sorted(relevances)[-2]  # second-highest
    pairs = store.similarity_search_with_relevance_scores(
        "alpha", k=5, score_threshold=cutoff
    )
    assert all(s >= cutoff for _, s in pairs)
    # At most two entries can be at-or-above the second-highest.
    assert len(pairs) <= 2


def test_score_threshold_returns_empty_when_nothing_qualifies():
    store = ChDBVectorStore(_DeterministicEmbedder(8), embedding_dimension=8)
    store.add_texts(["alpha", "beta"])
    pairs = store.similarity_search_with_relevance_scores(
        "alpha", k=5, score_threshold=1.5
    )
    assert pairs == []


async def test_async_score_threshold():
    store = ChDBVectorStore(_DeterministicEmbedder(8), embedding_dimension=8)
    await store.aadd_texts(["alpha", "beta", "gamma"])
    pairs = await store.asimilarity_search_with_relevance_scores(
        "alpha", k=3, score_threshold=2.0
    )
    assert pairs == []


def test_reopen_in_same_process_via_public_api_sees_prior_writes():
    """A second ``ChDBVectorStore`` against the same in-memory state
    must see existing rows through the public API alone — no private
    ``_ensure_table`` call required."""
    s1 = ChDBVectorStore(_DeterministicEmbedder(8), embedding_dimension=8)
    s1.add_texts(["alpha", "beta"], ids=["a", "b"])

    # Fresh instance against the same ``:memory:`` (shared per process).
    s2 = ChDBVectorStore(_DeterministicEmbedder(8), embedding_dimension=8)
    docs = s2.get_by_ids(["a", "b"])
    assert {d.id for d in docs} == {"a", "b"}

    hits = s2.similarity_search("alpha", k=5)
    assert {h.page_content for h in hits} == {"alpha", "beta"}


def test_relevance_mapping_inner_product_is_sigmoid():
    store = ChDBVectorStore(
        _DeterministicEmbedder(8),
        embedding_dimension=8,
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
    )
    assert math.isclose(store._raw_score_to_relevance(0.0), 0.5)
