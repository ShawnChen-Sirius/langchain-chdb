"""Shared pytest fixtures.

``chdb.session.Session(path=":memory:")`` is **process-global**: state
written from one ``Session`` instance is visible to every other
``Session`` instance in the same process, and survives ``close()``.
Tests that exercise the default in-memory database therefore see each
other's tables unless explicitly cleaned.

The autouse ``_isolate_inmemory_vectorstore_state`` fixture below drops
the default ``langchain_chdb_vectors`` table before each test so that
``ChDBVectorStore`` tests do not collide on the shared schema. The
fixture is a no-op for tests that don't touch the default table (e.g.
the loader tests), so adding it project-wide is safe.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_inmemory_vectorstore_state() -> None:
    """Sweep the default in-memory ``ChDBVectorStore`` table between tests."""
    from chdb.session import Session

    session = Session(path=":memory:")
    try:
        session.query("DROP TABLE IF EXISTS langchain_chdb_vectors")
    finally:
        session.close()
