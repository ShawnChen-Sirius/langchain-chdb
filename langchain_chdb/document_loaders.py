"""Document loader for chDB.

Executes a chDB SQL query and yields ``langchain_core.documents.Document``
objects, one per result row. Execution goes through
``chdb.session.Session.query(sql, "JSONEachRow")`` so that ``Array``,
``Map``, and ``JSON`` cells deserialize as native Python ``list`` / ``dict``
values rather than the ``repr()``-style strings that the chdb DB-API
cursor returns for the same columns.

The ``page_content_columns`` formatting strategy is documented in
``docs/decisions/loader_page_content_format.md``.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Iterator
from typing import Any

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document


class ChDBLoader(BaseLoader):
    """Load documents from a chDB SQL query result.

    Each row of the result becomes one ``Document``. The arguments
    ``page_content_columns`` and ``metadata_columns`` choose which row
    columns populate ``Document.page_content`` and ``Document.metadata``.

    Parameters
    ----------
    query:
        A chDB SELECT statement. Common shapes:

        * ``SELECT id, body FROM file('docs.parquet', 'Parquet')``
        * ``SELECT * FROM s3('s3://bucket/key.parquet')``
        * ``SELECT * FROM url('https://example.com/data.csv', 'CSV')``
        * ``SELECT * FROM remoteSecure('host:9440', 'db.tbl', ...)``
    database:
        Path to a persistent chDB store, or ``":memory:"`` (default) for a
        transient session. The loader opens and closes its own
        ``chdb.session.Session`` per ``load()`` / ``lazy_load()`` call;
        in-memory state from another ``Session`` is not visible here.
    page_content_columns:
        How ``Document.page_content`` is built from each row.

        * ``None`` (default): every selected column is serialized as
          one ``col: value`` line.
        * Exactly one entry (e.g. ``["body"]``): that column's raw cell
          value is used as ``Document.page_content`` directly. The
          recommended shape for RAG ingestion of a text body column.
        * Two or more entries (e.g. ``["title", "body"]``): only those
          columns are serialized, one per line, in the listed order.
    metadata_columns:
        How ``Document.metadata`` is populated.

        * ``None`` (default): every column not listed in
          ``page_content_columns`` becomes a metadata entry. If
          ``page_content_columns`` is also ``None``, ``metadata`` is
          empty — every column went into ``page_content``.
        * Explicit list: only those columns become metadata, regardless
          of what is in ``page_content_columns``.

    Notes
    -----
    Result rows are parsed from chDB's ``JSONEachRow`` output format, so
    ``Array(T)``, ``Map(K,V)``, ``Tuple(...)``, and ``JSON`` cells arrive
    as native Python types (list / dict / list / dict respectively) in
    ``Document.metadata``. The DB-API cursor in chdb 4.x returns those
    same types as Python ``repr()`` strings; this loader avoids that path
    deliberately.
    """

    def __init__(
        self,
        query: str,
        *,
        database: str = ":memory:",
        page_content_columns: list[str] | None = None,
        metadata_columns: list[str] | None = None,
    ) -> None:
        self.query = query
        self.database = database
        self.page_content_columns = page_content_columns
        self.metadata_columns = metadata_columns

    # ------------------------------------------------------------------
    # Sync interface
    # ------------------------------------------------------------------

    def lazy_load(self) -> Iterator[Document]:
        """Yield one ``Document`` per result row.

        ``chdb.session.Session.query`` materializes the full JSONEachRow
        response before returning, so this method does not stream in the
        memory-bounded sense — it iterates an already-in-memory result
        and yields ``Document`` objects one at a time. The interface is
        compatible with LangChain's ``BaseLoader.lazy_load`` contract.

        Raises
        ------
        ValueError
            If ``page_content_columns`` or ``metadata_columns`` references
            a column the query result does not contain.
        """
        from chdb.session import Session

        sess = Session(path=self.database)
        try:
            result = sess.query(self.query, "JSONEachRow")
            text = result if isinstance(result, str) else str(result)
            validated = False
            for line in text.splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                row: dict[str, Any] = json.loads(stripped)
                if not validated:
                    self._validate_requested_columns(row.keys())
                    validated = True
                yield self._row_to_document(row)
        finally:
            sess.close()

    def load(self) -> list[Document]:
        return list(self.lazy_load())

    # ------------------------------------------------------------------
    # Async interface
    # ------------------------------------------------------------------
    #
    # chDB is synchronous. We dispatch the blocking call to a worker
    # thread so the event loop is not stalled, then iterate the parsed
    # Documents back on the calling task. Behavior is identical to the
    # sync path; only the scheduling differs. ``alazy_load`` materializes
    # the full list inside the worker thread before yielding — it is not
    # memory-bounded.

    async def aload(self) -> list[Document]:
        return await asyncio.to_thread(self.load)

    async def alazy_load(self) -> AsyncIterator[Document]:
        docs = await asyncio.to_thread(self.load)
        for doc in docs:
            yield doc

    # ------------------------------------------------------------------
    # Column validation
    # ------------------------------------------------------------------

    def _validate_requested_columns(self, available: Any) -> None:
        """Raise ``ValueError`` if any requested column is missing.

        Called once on the first result row. If the query returns zero
        rows there is nothing to validate against; bad column names in
        that case surface as an empty document list rather than an
        error.
        """
        available_set = set(available)
        for arg_name, requested in (
            ("page_content_columns", self.page_content_columns),
            ("metadata_columns", self.metadata_columns),
        ):
            if requested is None:
                continue
            missing = [c for c in requested if c not in available_set]
            if missing:
                raise ValueError(
                    f"{arg_name} references columns not in the query result: "
                    f"{missing!r}. Available columns: {sorted(available_set)!r}."
                )

    # ------------------------------------------------------------------
    # Row -> Document construction
    # ------------------------------------------------------------------

    def _row_to_document(self, row: dict[str, Any]) -> Document:
        return Document(
            page_content=self._build_page_content(row),
            metadata=self._build_metadata(row),
        )

    def _build_page_content(self, row: dict[str, Any]) -> str:
        if self.page_content_columns is None:
            return "\n".join(f"{k}: {v}" for k, v in row.items())
        if len(self.page_content_columns) == 1:
            col = self.page_content_columns[0]
            value = row.get(col, "")
            return value if isinstance(value, str) else str(value)
        return "\n".join(
            f"{c}: {row[c]}" for c in self.page_content_columns if c in row
        )

    def _build_metadata(self, row: dict[str, Any]) -> dict[str, Any]:
        if self.metadata_columns is not None:
            return {c: row[c] for c in self.metadata_columns if c in row}
        if self.page_content_columns is None:
            return {}
        return {k: v for k, v in row.items() if k not in self.page_content_columns}
