"""LangChain ``VectorStoreIntegrationTests`` conformance for ``ChDBVectorStore``.

This subclass plugs ``ChDBVectorStore`` into the upstream
``langchain_tests.integration_tests.vectorstores.VectorStoreIntegrationTests``
suite. The plan calls this the gating event for Step 3 — when this file's
tests pass, the contract is satisfied.

The suite has no dependency on ``[test-integration]`` extras (HF embedder
is not used); it exercises a built-in ``DeterministicFakeEmbedding`` so
runs on default CI.
"""

from __future__ import annotations

from collections.abc import Generator

import pytest
from langchain_core.vectorstores import VectorStore
from langchain_tests.integration_tests.vectorstores import VectorStoreIntegrationTests

from langchain_chdb import ChDBVectorStore


class TestChDBVectorStoreConformance(VectorStoreIntegrationTests):
    """LangChain conformance for ``ChDBVectorStore``.

    The autouse ``_isolate_inmemory_vectorstore_state`` fixture in
    ``conftest.py`` already drops the default backing table between
    tests, so each test method receives a clean store.
    """

    @pytest.fixture
    def vectorstore(self) -> Generator[VectorStore, None, None]:
        # ``DeterministicFakeEmbedding`` from langchain-tests produces
        # 6-dim vectors; declare the dimension explicitly so the chDB
        # CHECK constraint is set up before the first write.
        store = ChDBVectorStore(
            embedding=self.get_embeddings(),
            embedding_dimension=6,
        )
        try:
            yield store
        finally:
            store.close()
