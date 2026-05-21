"""Tests for ``ChDBLoader`` (Step 2 of the v0.1 execution plan).

Coverage:

* Single-column ``page_content_columns`` returns the raw cell value.
* Multi-column ``page_content_columns`` produces ``col: value`` lines.
* Default ``page_content_columns=None`` serializes every selected column.
* ``metadata_columns`` default = "every column not in page_content_columns".
* ``metadata_columns`` explicit = exactly the listed columns.
* Empty query result returns ``[]``.
* ``Array``-typed cells deserialize as native Python ``list`` (regression
  for the chdb DB-API ``repr()``-string return path; see plan D7).
* Async (``aload`` / ``alazy_load``) parity with sync.
"""

from __future__ import annotations

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from langchain_chdb import ChDBLoader

# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def docs_parquet(tmp_path):
    """Three-column tabular fixture (id, body, src)."""
    p = tmp_path / "docs.parquet"
    df = pd.DataFrame(
        {
            "id": [1, 2],
            "body": ["hello world", "foo bar"],
            "src": ["a", "b"],
        }
    )
    df.to_parquet(p)
    return p


@pytest.fixture
def array_parquet(tmp_path):
    """Single-row fixture with an Array(String) column."""
    table = pa.table(
        {
            "id": pa.array([1]),
            "tags": pa.array([["x", "y", "z"]]),
        }
    )
    p = tmp_path / "arr.parquet"
    pq.write_table(table, p)
    return p


# ---------------------------------------------------------------------------
# page_content shape (per D4)
# ---------------------------------------------------------------------------


def test_single_column_returns_raw_content(docs_parquet):
    loader = ChDBLoader(
        query=f"SELECT id, body, src FROM file('{docs_parquet}', 'Parquet') ORDER BY id",
        page_content_columns=["body"],
        metadata_columns=["id"],
    )
    docs = loader.load()
    assert len(docs) == 2
    assert docs[0].page_content == "hello world"
    assert docs[0].metadata == {"id": 1}
    assert docs[1].page_content == "foo bar"
    assert docs[1].metadata == {"id": 2}


def test_multi_column_uses_key_value_format(docs_parquet):
    loader = ChDBLoader(
        query=f"SELECT id, body, src FROM file('{docs_parquet}', 'Parquet') ORDER BY id",
        page_content_columns=["body", "src"],
        metadata_columns=["id"],
    )
    docs = loader.load()
    assert docs[0].page_content == "body: hello world\nsrc: a"
    assert docs[1].page_content == "body: foo bar\nsrc: b"


def test_default_concats_all_columns(docs_parquet):
    loader = ChDBLoader(
        query=f"SELECT id, body, src FROM file('{docs_parquet}', 'Parquet') ORDER BY id",
    )
    docs = loader.load()
    assert "id: 1" in docs[0].page_content
    assert "body: hello world" in docs[0].page_content
    assert "src: a" in docs[0].page_content
    # default: page_content has everything, metadata is empty
    assert docs[0].metadata == {}


# ---------------------------------------------------------------------------
# metadata shape
# ---------------------------------------------------------------------------


def test_metadata_default_excludes_page_content_columns(docs_parquet):
    loader = ChDBLoader(
        query=f"SELECT id, body, src FROM file('{docs_parquet}', 'Parquet') ORDER BY id",
        page_content_columns=["body"],
    )
    docs = loader.load()
    assert docs[0].metadata == {"id": 1, "src": "a"}


def test_metadata_columns_explicit_when_page_content_columns_none(docs_parquet):
    loader = ChDBLoader(
        query=f"SELECT id, body, src FROM file('{docs_parquet}', 'Parquet') ORDER BY id",
        metadata_columns=["id", "src"],
    )
    docs = loader.load()
    assert docs[0].metadata == {"id": 1, "src": "a"}


# ---------------------------------------------------------------------------
# edge cases
# ---------------------------------------------------------------------------


def test_empty_result_returns_empty_list(tmp_path):
    p = tmp_path / "empty.parquet"
    pd.DataFrame({"id": [1, 2]}).to_parquet(p)
    loader = ChDBLoader(
        query=f"SELECT id FROM file('{p}', 'Parquet') WHERE id < 0",
    )
    assert loader.load() == []


def test_array_column_preserves_list_type(array_parquet):
    """Regression for plan D7: chdb.dbapi serializes Array cells as
    ``repr()`` strings, but JSONEachRow output gives a native list.
    ``ChDBLoader`` must route through JSONEachRow for hot paths."""
    loader = ChDBLoader(
        query=f"SELECT id, tags FROM file('{array_parquet}', 'Parquet')",
        page_content_columns=["id"],
        metadata_columns=["tags"],
    )
    docs = loader.load()
    assert len(docs) == 1
    assert isinstance(docs[0].metadata["tags"], list)
    assert docs[0].metadata["tags"] == ["x", "y", "z"]


# ---------------------------------------------------------------------------
# async parity
# ---------------------------------------------------------------------------


async def test_aload_matches_load(docs_parquet):
    loader = ChDBLoader(
        query=f"SELECT id, body FROM file('{docs_parquet}', 'Parquet') ORDER BY id",
        page_content_columns=["body"],
        metadata_columns=["id"],
    )
    sync_docs = loader.load()
    async_docs = await loader.aload()
    assert len(sync_docs) == len(async_docs)
    for s, a in zip(sync_docs, async_docs, strict=True):
        assert s.page_content == a.page_content
        assert s.metadata == a.metadata


async def test_alazy_load_yields_documents(docs_parquet):
    loader = ChDBLoader(
        query=f"SELECT id, body FROM file('{docs_parquet}', 'Parquet') ORDER BY id",
        page_content_columns=["body"],
    )
    yielded = [doc async for doc in loader.alazy_load()]
    assert len(yielded) == 2
    assert yielded[0].page_content == "hello world"
    assert yielded[1].page_content == "foo bar"
