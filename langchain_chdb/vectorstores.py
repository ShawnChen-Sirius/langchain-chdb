"""ChDBVectorStore — a LangChain vector store backed by chDB.

Storage model
-------------

Each store maps to one chDB table::

    CREATE TABLE {table_name} (
        id        String,
        content   String,
        metadata  JSON,
        embedding Array(Float32),
        CONSTRAINT embedding_dim_check CHECK length(embedding) = {N}
    )
    ENGINE = MergeTree()
    ORDER BY id;

``ORDER BY id`` is *not* a uniqueness constraint in ClickHouse — same id
can appear in multiple rows. ``add_documents`` therefore implements upsert
as ``DELETE WHERE id IN (...) SETTINGS mutations_sync = 1`` followed by
``INSERT``, buying read-after-write determinism at the cost of one
synchronous mutation per write. v0.2 is expected to migrate to an
append-only schema with query-side dedup.

Search uses the chDB scalar distance functions:

* ``DistanceStrategy.COSINE``       → ``cosineDistance(embedding, q)``  smaller = closer
* ``DistanceStrategy.EUCLIDEAN``    → ``L2Distance(embedding, q)``      smaller = closer
* ``DistanceStrategy.MAX_INNER_PRODUCT`` → ``dotProduct(embedding, q)`` larger  = closer

``similarity_search_with_score`` returns the raw chDB value (distance for
the first two strategies, inner-product for the third).
``similarity_search_with_relevance_scores`` maps every strategy's raw value
into the ``[0, 1]`` interval, monotone in semantic closeness, so callers
can compare across strategies.

Metadata filter DSL
-------------------

Filters are an explicit whitelist; unknown operators raise ``ValueError``
before any SQL is emitted::

    {"key": "value"}                            # equality
    {"key": {"$in": ["a", "b"]}}                # membership (OR of eqs)
    {"key": {"$gt": 0.5}}                       # comparison ($gt/$gte/$lt/$lte)
    {"key": {"$ne": "x"}}                       # inequality
    {"$and": [filter1, filter2, ...]}           # conjunction
    {"$or":  [filter1, filter2, ...]}           # disjunction
    {"$not": filter}                            # negation

Metadata keys must match ``[A-Za-z_][A-Za-z0-9_]*``. Users who need
non-identifier keys can issue a raw SQL filter outside this DSL.

Result-path note
----------------

chDB's DB-API cursor serializes ``Array(T)`` / ``Map(K,V)`` / ``Tuple(...)``
/ ``JSON`` cells as Python ``repr()`` strings rather than native objects.
This module dispatches every read through ``Session.query(sql,
"JSONEachRow")`` so cells arrive as the right Python types from
``json.loads``.
"""

from __future__ import annotations

import asyncio
import contextlib
import enum
import json
import re
import uuid
from collections.abc import Iterable
from typing import Any, ClassVar

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from langchain_chdb._sql import quote_identifier as _quote_identifier

# ---------------------------------------------------------------------------
# Distance strategies
# ---------------------------------------------------------------------------


class DistanceStrategy(str, enum.Enum):
    """Distance strategy for vector similarity in ``ChDBVectorStore``."""

    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    MAX_INNER_PRODUCT = "max_inner_product"


# function name in chDB, and whether smaller-is-closer (True) or larger-is-closer (False).
_DISTANCE_FN: dict[DistanceStrategy, tuple[str, bool]] = {
    DistanceStrategy.COSINE: ("cosineDistance", True),
    DistanceStrategy.EUCLIDEAN: ("L2Distance", True),
    DistanceStrategy.MAX_INNER_PRODUCT: ("dotProduct", False),
}


# ---------------------------------------------------------------------------
# SQL escaping helpers
# ---------------------------------------------------------------------------


_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _bind(params: dict[str, Any], value: Any) -> str:
    """Bind a scalar as a chDB param and return its ``{name:Type}`` placeholder.

    Values never reach the SQL text — they are sent as server-side parameters.
    ``None`` is the one exception (rendered as the ``NULL`` literal, which cannot
    carry a value). Booleans bind as 0/1.
    """
    if value is None:
        return "NULL"
    name = f"p{len(params)}"
    if isinstance(value, bool):
        params[name] = 1 if value else 0
        return f"{{{name}:UInt8}}"
    if isinstance(value, int):
        params[name] = value
        return f"{{{name}:Int64}}"
    if isinstance(value, float):
        params[name] = value
        return f"{{{name}:Float64}}"
    if isinstance(value, str):
        params[name] = value
        return f"{{{name}:String}}"
    raise TypeError(
        f"unsupported value type for filter: {type(value).__name__} "
        f"({value!r}). Allowed: str, int, float, bool, None."
    )


def _format_embedding_literal(vector: list[float]) -> str:
    """Render a float vector as a chDB ``Array(Float32)`` literal.

    Floats are not an injection vector, so the vector is inlined directly.
    """
    return "[" + ",".join(repr(float(v)) for v in vector) + "]"


# ---------------------------------------------------------------------------
# Filter DSL → SQL
# ---------------------------------------------------------------------------


_COMPARISON_OPS: dict[str, str] = {
    "$gt": ">",
    "$gte": ">=",
    "$lt": "<",
    "$lte": "<=",
    "$ne": "!=",
}

_LOGICAL_OPS: frozenset[str] = frozenset({"$and", "$or", "$not"})


def _json_path_expr(metadata_column: str, key: str) -> str:
    """Render the SQL expression for a JSON path lookup, e.g. ``metadata.category``."""
    if not _IDENTIFIER_RE.match(key):
        raise ValueError(
            f"Invalid metadata key {key!r}: must match [A-Za-z_][A-Za-z0-9_]*"
        )
    return f"{_quote_identifier(metadata_column)}.{key}"


def _equality_clause(metadata_column: str, key: str, value: Any, params: dict[str, Any]) -> str:
    path = _json_path_expr(metadata_column, key)
    if isinstance(value, str):
        # Cast both sides to String to avoid the Dynamic-type IN/comparison
        # mismatches on chDB's typed JSON path.
        return f"toString({path}) = {_bind(params, value)}"
    return f"{path} = {_bind(params, value)}"


def _comparison_clause(metadata_column: str, key: str, op: str, value: Any, params: dict[str, Any]) -> str:
    path = _json_path_expr(metadata_column, key)
    return f"{path} {_COMPARISON_OPS[op]} {_bind(params, value)}"


def _field_dict_clause(metadata_column: str, key: str, op_dict: dict[str, Any], params: dict[str, Any]) -> str:
    parts: list[str] = []
    for op, op_val in op_dict.items():
        if op == "$in":
            if not isinstance(op_val, list) or not op_val:
                raise ValueError(f"$in requires a non-empty list, got {op_val!r}")
            sub = [_equality_clause(metadata_column, key, v, params) for v in op_val]
            parts.append("(" + " OR ".join(sub) + ")")
        elif op == "$ne" and isinstance(op_val, str):
            path = _json_path_expr(metadata_column, key)
            parts.append(f"toString({path}) != {_bind(params, op_val)}")
        elif op in _COMPARISON_OPS:
            parts.append(_comparison_clause(metadata_column, key, op, op_val, params))
        else:
            raise ValueError(
                f"Unsupported filter operator {op!r} on field {key!r}. "
                f"Allowed: $in, $gt, $gte, $lt, $lte, $ne."
            )
    return " AND ".join(parts) if len(parts) > 1 else parts[0]


def _filter_to_sql(filter_obj: dict[str, Any], metadata_column: str, params: dict[str, Any]) -> str:
    """Compile a filter DSL dict into a chDB SQL boolean expression.

    Filter *values* are appended to ``params`` and referenced as bound
    parameters — never interpolated. Raises ``ValueError`` for any unknown
    operator. Empty filter returns the empty string; the caller decides whether
    to emit a ``WHERE`` at all.
    """
    if not isinstance(filter_obj, dict):
        raise ValueError(
            f"Filter must be a dict, got {type(filter_obj).__name__}: {filter_obj!r}"
        )
    if not filter_obj:
        return ""

    parts: list[str] = []
    for key, value in filter_obj.items():
        if key == "$and":
            if not isinstance(value, list):
                raise ValueError(f"$and requires a list of sub-filters, got {value!r}")
            sub_clauses = [_filter_to_sql(f, metadata_column, params) for f in value]
            parts.append("(" + " AND ".join(c for c in sub_clauses if c) + ")")
        elif key == "$or":
            if not isinstance(value, list):
                raise ValueError(f"$or requires a list of sub-filters, got {value!r}")
            sub_clauses = [_filter_to_sql(f, metadata_column, params) for f in value]
            parts.append("(" + " OR ".join(c for c in sub_clauses if c) + ")")
        elif key == "$not":
            sub_clause = _filter_to_sql(value, metadata_column, params)
            parts.append(f"NOT ({sub_clause})")
        elif key.startswith("$"):
            raise ValueError(
                f"Unsupported top-level operator {key!r}. "
                f"Allowed: $and, $or, $not."
            )
        elif isinstance(value, dict):
            parts.append(_field_dict_clause(metadata_column, key, value, params))
        else:
            parts.append(_equality_clause(metadata_column, key, value, params))

    return " AND ".join(parts)


# ---------------------------------------------------------------------------
# The store itself
# ---------------------------------------------------------------------------


class ChDBVectorStore(VectorStore):
    """A LangChain ``VectorStore`` backed by a single chDB table.

    Parameters
    ----------
    embedding:
        Any LangChain ``Embeddings`` implementation. The store calls
        ``embed_documents`` on writes and ``embed_query`` on searches.
    database:
        Path to a persistent chDB store, or ``":memory:"`` for a
        transient session. The store opens its own ``chdb.session.Session``
        and keeps it alive for the lifetime of the instance.
    table_name:
        Backing table name. Must match ``[A-Za-z_][A-Za-z0-9_]*``.
    embedding_dimension:
        Required vector length. If ``None`` at construction, the first
        ``add_*`` call sets it from the produced embeddings. Once set,
        every subsequent embedding must match — mismatches raise
        ``ValueError`` before any SQL runs, and the chDB
        ``CHECK length(embedding) = N`` constraint catches anything
        that slipped through.
    content_column, metadata_column, id_column:
        Column names for the three first-class fields. The embedding
        column is always ``"embedding"`` in v0.1.
    distance_strategy:
        Default similarity function. Override per call via the
        ``similarity_search`` argument once exposed (v0.1 uses this as
        the only setting).
    create_if_not_exists:
        When ``True`` (default), the store creates the backing table on
        first write if it doesn't exist. When ``False``, the table must
        already exist with a matching schema.
    readonly:
        When ``True``, all write methods raise ``RuntimeError``. Useful
        for serving a pre-built store to untrusted code.
    """

    EMBEDDING_COLUMN: ClassVar[str] = "embedding"

    def __init__(
        self,
        embedding: Embeddings,
        *,
        database: str = ":memory:",
        table_name: str = "langchain_chdb_vectors",
        embedding_dimension: int | None = None,
        content_column: str = "content",
        metadata_column: str = "metadata",
        id_column: str = "id",
        distance_strategy: DistanceStrategy = DistanceStrategy.COSINE,
        create_if_not_exists: bool = True,
        readonly: bool = False,
    ) -> None:
        # validate identifiers up front
        _quote_identifier(table_name)
        _quote_identifier(content_column)
        _quote_identifier(metadata_column)
        _quote_identifier(id_column)

        if embedding_dimension is not None and embedding_dimension <= 0:
            raise ValueError(
                f"embedding_dimension must be positive, got {embedding_dimension!r}"
            )
        if not isinstance(distance_strategy, DistanceStrategy):
            raise TypeError(
                "distance_strategy must be a DistanceStrategy member, "
                f"got {type(distance_strategy).__name__}"
            )

        self._embedding = embedding
        self._database = database
        self._table_name = table_name
        self._embedding_dimension = embedding_dimension
        self._content_column = content_column
        self._metadata_column = metadata_column
        self._id_column = id_column
        self._distance_strategy = distance_strategy
        self._create_if_not_exists = create_if_not_exists
        self._readonly = readonly

        self._session: Any = None
        self._table_initialized = False

    # ------------------------------------------------------------------
    # session / lifecycle
    # ------------------------------------------------------------------

    def _get_session(self) -> Any:
        if self._session is None:
            from chdb.session import Session
            self._session = Session(path=self._database)
        return self._session

    def close(self) -> None:
        """Close the underlying ``chdb.session.Session``.

        After ``close()``, every method raises. Reusable via a new
        ``ChDBVectorStore`` instance against the same ``database``.
        """
        if self._session is not None:
            self._session.close()
            self._session = None
            self._table_initialized = False

    def __del__(self) -> None:
        # never raise from __del__
        with contextlib.suppress(Exception):
            self.close()

    @property
    def embeddings(self) -> Embeddings:
        return self._embedding

    # ------------------------------------------------------------------
    # DDL
    # ------------------------------------------------------------------

    def _table_exists_in_chdb(self) -> bool:
        """Probe whether the backing table exists in the bound database.

        Used by read methods so a freshly-instantiated store against an
        existing on-disk database returns real rows instead of ``[]``.
        """
        sql = (
            "SELECT 1 FROM system.tables "
            "WHERE database = currentDatabase() "
            "AND name = {tbl:String} LIMIT 1"
        )
        return bool(self._query_jsoneachrow(sql, {"tbl": self._table_name}))

    def _ready_for_read(self) -> bool:
        """Mark the table as initialized on first read against an existing
        on-disk table; return False if the table truly doesn't exist."""
        if self._table_initialized:
            return True
        if self._table_exists_in_chdb():
            self._table_initialized = True
            return True
        return False

    def _ensure_table(self, embedding_dim: int) -> None:
        """Lazily create the backing table on the first write.

        Records the embedding dimension on the instance the first time it
        runs. If the user passed ``embedding_dimension`` at construction
        and the inferred dim disagrees, raises ``ValueError`` before any
        DDL.
        """
        if self._table_initialized:
            return
        if self._embedding_dimension is None:
            self._embedding_dimension = embedding_dim
        elif self._embedding_dimension != embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: declared "
                f"{self._embedding_dimension}, got {embedding_dim} from "
                f"the embedder."
            )

        if not self._create_if_not_exists:
            self._table_initialized = True
            return

        ddl = (
            f"CREATE TABLE IF NOT EXISTS {_quote_identifier(self._table_name)} (\n"
            f"    {_quote_identifier(self._id_column)} String,\n"
            f"    {_quote_identifier(self._content_column)} String,\n"
            f"    {_quote_identifier(self._metadata_column)} JSON,\n"
            f"    {_quote_identifier(self.EMBEDDING_COLUMN)} Array(Float32),\n"
            f"    CONSTRAINT embedding_dim_check CHECK "
            f"length({_quote_identifier(self.EMBEDDING_COLUMN)}) "
            f"= {self._embedding_dimension}\n"
            f") ENGINE = MergeTree() ORDER BY {_quote_identifier(self._id_column)}"
        )
        self._get_session().query(ddl)
        self._table_initialized = True

    # ------------------------------------------------------------------
    # write path — DELETE+INSERT upsert
    # ------------------------------------------------------------------

    def _resolve_ids(
        self, documents: list[Document], ids: list[str] | None
    ) -> list[str]:
        """Pick an id for each document without mutating the input.

        LangChain's ``VectorStoreIntegrationTests`` contract forbids
        ``add_documents`` from writing back onto the caller's ``Document``
        instances. ``Document.id`` is read when present and ignored
        otherwise; the resolved id list is returned alongside the
        upserted rows.
        """
        if ids is not None:
            if len(ids) != len(documents):
                raise ValueError(
                    f"ids length ({len(ids)}) does not match documents "
                    f"length ({len(documents)})"
                )
            return list(ids)
        return [
            doc.id
            if (doc.id is not None and isinstance(doc.id, str) and doc.id)
            else str(uuid.uuid4())
            for doc in documents
        ]

    @staticmethod
    def _fold_batch_by_id(
        ids: list[str],
        documents: list[Document],
        embeddings: list[list[float]],
    ) -> tuple[list[str], list[Document], list[list[float]]]:
        """Collapse same-id rows in a single batch, keeping the last write.

        ``add_documents([Document(id="1", page_content="a"),
        Document(id="1", page_content="b")])`` must result in exactly
        one row with content ``"b"`` — without this fold the upsert
        would ``DELETE WHERE id IN ('1')`` once and then ``INSERT``
        both rows, leaving the physical table with two id-equal rows
        even if ``get_by_ids`` happened to fold them away by dict.
        """
        last_index: dict[str, int] = {}
        for i, doc_id in enumerate(ids):
            last_index[doc_id] = i
        kept = sorted(last_index.values())
        return (
            [ids[i] for i in kept],
            [documents[i] for i in kept],
            [embeddings[i] for i in kept],
        )

    def _validate_embeddings(self, embeddings: list[list[float]]) -> None:
        if not embeddings:
            return
        expected = self._embedding_dimension
        if expected is None:
            return
        for i, vec in enumerate(embeddings):
            if len(vec) != expected:
                raise ValueError(
                    f"Embedding at index {i} has length {len(vec)}, "
                    f"expected {expected}"
                )

    def _sync_delete_by_ids(self, ids: Iterable[str]) -> None:
        id_list = list(ids)
        if not id_list:
            return
        params: dict[str, Any] = {}
        in_clause = ", ".join(_bind(params, i) for i in id_list)
        self._get_session().query(
            f"ALTER TABLE {_quote_identifier(self._table_name)} "
            f"DELETE WHERE {_quote_identifier(self._id_column)} IN ({in_clause}) "
            f"SETTINGS mutations_sync = 1",
            params=params,
        )

    def _insert_rows(
        self,
        ids: list[str],
        documents: list[Document],
        embeddings: list[list[float]],
    ) -> None:
        # id / content / metadata are bound per row; the embedding is a float
        # vector (not injectable) and is inlined as an Array(Float32) literal.
        params: dict[str, Any] = {}
        rows: list[str] = []
        for doc_id, doc, vec in zip(ids, documents, embeddings, strict=True):
            meta_json = json.dumps(doc.metadata or {}, ensure_ascii=False, separators=(",", ":"))
            rows.append(
                "("
                f"{_bind(params, doc_id)}, "
                f"{_bind(params, doc.page_content)}, "
                f"{_bind(params, meta_json)}, "
                f"{_format_embedding_literal(vec)}"
                ")"
            )
        self._get_session().query(
            f"INSERT INTO {_quote_identifier(self._table_name)} "
            f"({_quote_identifier(self._id_column)}, "
            f"{_quote_identifier(self._content_column)}, "
            f"{_quote_identifier(self._metadata_column)}, "
            f"{_quote_identifier(self.EMBEDDING_COLUMN)}) "
            f"VALUES " + ",\n".join(rows),
            params=params,
        )

    def _require_writable(self) -> None:
        if self._readonly:
            raise RuntimeError(
                "ChDBVectorStore is readonly; write methods are disabled."
            )

    def add_documents(
        self,
        documents: list[Document],
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Upsert a batch of documents.

        Returns the list of stored IDs in input order. Re-adding a
        document with an id that already exists overwrites the prior
        row — the upsert is implemented as ``DELETE WHERE id IN (...)
        SETTINGS mutations_sync = 1`` followed by ``INSERT``.
        """
        self._require_writable()
        if not documents:
            return []

        resolved_ids = self._resolve_ids(documents, ids)
        raw_embeddings = self._embedding.embed_documents(
            [d.page_content for d in documents]
        )
        embeddings = [list(map(float, v)) for v in raw_embeddings]

        if not embeddings or not embeddings[0]:
            raise ValueError("Embedder returned empty embeddings")

        # Fold same-id rows in this batch *before* the embedder check so
        # the dim contract is enforced on what actually gets written.
        folded_ids, folded_docs, folded_embs = self._fold_batch_by_id(
            resolved_ids, documents, embeddings
        )

        self._ensure_table(embedding_dim=len(folded_embs[0]))
        self._validate_embeddings(folded_embs)
        self._sync_delete_by_ids(folded_ids)
        self._insert_rows(folded_ids, folded_docs, folded_embs)
        # Return the ids the caller passed in (preserving order and
        # duplicates) — they map 1:1 to input documents even though the
        # physical write collapsed.
        return resolved_ids

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict[str, Any]] | None = None,
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        text_list = list(texts)
        meta_list = list(metadatas) if metadatas is not None else [{}] * len(text_list)
        if len(meta_list) != len(text_list):
            raise ValueError(
                f"metadatas length ({len(meta_list)}) does not match texts "
                f"length ({len(text_list)})"
            )
        documents = [
            Document(page_content=t, metadata=m or {})
            for t, m in zip(text_list, meta_list, strict=True)
        ]
        return self.add_documents(documents, ids=ids)

    # ------------------------------------------------------------------
    # delete
    # ------------------------------------------------------------------

    def delete(self, ids: list[str] | None = None, **kwargs: Any) -> bool | None:
        """Synchronously delete rows by id; returns True on success.

        The ``mutations_sync = 1`` setting makes the deletion visible to
        subsequent reads from the same session immediately.
        """
        self._require_writable()
        if ids is None:
            return None
        if not ids:
            return True
        # Ensure the table exists before issuing a DELETE; on a freshly-
        # constructed store with no prior writes, deleting from a
        # non-existent table would error.
        if not self._table_initialized and self._create_if_not_exists:
            # Lazy DDL needs a dimension. If we have none declared and no
            # prior write, there's nothing to delete from — return True.
            if self._embedding_dimension is None:
                return True
            self._ensure_table(embedding_dim=self._embedding_dimension)
        self._sync_delete_by_ids(ids)
        return True

    # ------------------------------------------------------------------
    # get_by_ids
    # ------------------------------------------------------------------

    def get_by_ids(self, ids: list[str]) -> list[Document]:
        """Fetch documents by id. Missing ids are simply omitted.

        Order of returned documents follows the order of *requested* ids
        — duplicates in the request return duplicate documents, missing
        ids are skipped.
        """
        if not ids:
            return []
        if not self._ready_for_read():
            return []

        params: dict[str, Any] = {}
        in_clause = ", ".join(_bind(params, i) for i in ids)
        sql = (
            f"SELECT {_quote_identifier(self._id_column)} AS id, "
            f"{_quote_identifier(self._content_column)} AS content, "
            f"{_quote_identifier(self._metadata_column)} AS metadata "
            f"FROM {_quote_identifier(self._table_name)} "
            f"WHERE {_quote_identifier(self._id_column)} IN ({in_clause})"
        )
        rows = self._query_jsoneachrow(sql, params)
        by_id: dict[str, Document] = {}
        for row in rows:
            by_id[row["id"]] = Document(
                id=row["id"],
                page_content=row["content"],
                metadata=row.get("metadata") or {},
            )
        return [by_id[i] for i in ids if i in by_id]

    # ------------------------------------------------------------------
    # similarity search
    # ------------------------------------------------------------------

    def similarity_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        *,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        return [doc for doc, _ in self.similarity_search_by_vector_with_score(
            embedding, k=k, filter=filter
        )]

    def similarity_search_by_vector_with_score(
        self,
        embedding: list[float],
        k: int = 4,
        *,
        filter: dict[str, Any] | None = None,
    ) -> list[tuple[Document, float]]:
        if self._embedding_dimension is not None and len(embedding) != self._embedding_dimension:
            raise ValueError(
                f"Query embedding has length {len(embedding)}, "
                f"expected {self._embedding_dimension}"
            )
        if not self._ready_for_read():
            return []

        fn, _smaller_is_closer = _DISTANCE_FN[self._distance_strategy]
        order = "ASC" if _smaller_is_closer else "DESC"
        emb_col = _quote_identifier(self.EMBEDDING_COLUMN)
        q_lit = _format_embedding_literal(embedding)

        params: dict[str, Any] = {}
        where_sql = ""
        if filter:
            clause = _filter_to_sql(filter, self._metadata_column, params)
            if clause:
                where_sql = f"WHERE {clause} "

        sql = (
            f"SELECT {_quote_identifier(self._id_column)} AS id, "
            f"{_quote_identifier(self._content_column)} AS content, "
            f"{_quote_identifier(self._metadata_column)} AS metadata, "
            f"{fn}({emb_col}, {q_lit}) AS score "
            f"FROM {_quote_identifier(self._table_name)} "
            f"{where_sql}"
            f"ORDER BY score {order} "
            f"LIMIT {int(k)}"
        )
        rows = self._query_jsoneachrow(sql, params)
        return [
            (
                Document(
                    id=row["id"],
                    page_content=row["content"],
                    metadata=row.get("metadata") or {},
                ),
                float(row["score"]),
            )
            for row in rows
        ]

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        *,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        return [
            doc for doc, _ in self.similarity_search_with_score(query, k=k, filter=filter)
        ]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        *,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        vec = list(map(float, self._embedding.embed_query(query)))
        return self.similarity_search_by_vector_with_score(vec, k=k, filter=filter)

    def similarity_search_with_relevance_scores(
        self,
        query: str,
        k: int = 4,
        *,
        filter: dict[str, Any] | None = None,
        score_threshold: float | None = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Return ``(Document, relevance)`` pairs, optionally filtered by threshold.

        ``score_threshold`` is applied to the relevance score (the
        ``[0, 1]`` value), not the raw chDB distance. Pairs with
        relevance below the threshold are dropped from the result.
        """
        raw = self.similarity_search_with_score(query, k=k, filter=filter)
        pairs = [(doc, self._raw_score_to_relevance(score)) for doc, score in raw]
        if score_threshold is not None:
            pairs = [(d, s) for d, s in pairs if s >= score_threshold]
        return pairs

    def _raw_score_to_relevance(self, score: float) -> float:
        """Map a strategy-specific raw score into the ``[0, 1]`` interval."""
        s = self._distance_strategy
        if s is DistanceStrategy.COSINE:
            # Cosine distance is in [0, 2]; for unit-norm embeddings it
            # tends to land in [0, 1] but we clamp defensively.
            return max(0.0, min(1.0, 1.0 - float(score)))
        if s is DistanceStrategy.EUCLIDEAN:
            return 1.0 / (1.0 + float(score))
        # MAX_INNER_PRODUCT — sigmoid maps the unbounded score into (0, 1).
        import math
        return 1.0 / (1.0 + math.exp(-float(score)))

    # ------------------------------------------------------------------
    # classmethod constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict[str, Any]] | None = None,
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> ChDBVectorStore:
        store = cls(embedding=embedding, **kwargs)
        if texts:
            store.add_texts(texts, metadatas=metadatas, ids=ids)
        return store

    @classmethod
    def from_documents(
        cls,
        documents: list[Document],
        embedding: Embeddings,
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> ChDBVectorStore:
        store = cls(embedding=embedding, **kwargs)
        if documents:
            store.add_documents(documents, ids=ids)
        return store

    # ------------------------------------------------------------------
    # async wrappers
    # ------------------------------------------------------------------

    async def aadd_documents(
        self,
        documents: list[Document],
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        return await asyncio.to_thread(self.add_documents, documents, ids=ids)

    async def aadd_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict[str, Any]] | None = None,
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        return await asyncio.to_thread(
            self.add_texts, texts, metadatas=metadatas, ids=ids
        )

    async def aget_by_ids(self, ids: list[str]) -> list[Document]:
        return await asyncio.to_thread(self.get_by_ids, ids)

    async def adelete(self, ids: list[str] | None = None, **kwargs: Any) -> bool | None:
        return await asyncio.to_thread(self.delete, ids)

    async def asimilarity_search(
        self,
        query: str,
        k: int = 4,
        *,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        return await asyncio.to_thread(self.similarity_search, query, k, filter=filter)

    async def asimilarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        *,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        return await asyncio.to_thread(
            self.similarity_search_with_score, query, k, filter=filter
        )

    async def asimilarity_search_with_relevance_scores(
        self,
        query: str,
        k: int = 4,
        *,
        filter: dict[str, Any] | None = None,
        score_threshold: float | None = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        return await asyncio.to_thread(
            self.similarity_search_with_relevance_scores,
            query,
            k,
            filter=filter,
            score_threshold=score_threshold,
        )

    async def asimilarity_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        *,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        return await asyncio.to_thread(
            self.similarity_search_by_vector, embedding, k, filter=filter
        )

    @classmethod
    async def afrom_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict[str, Any]] | None = None,
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> ChDBVectorStore:
        return await asyncio.to_thread(
            cls.from_texts, texts, embedding, metadatas, ids=ids, **kwargs
        )

    @classmethod
    async def afrom_documents(
        cls,
        documents: list[Document],
        embedding: Embeddings,
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> ChDBVectorStore:
        return await asyncio.to_thread(
            cls.from_documents, documents, embedding, ids=ids, **kwargs
        )

    # ------------------------------------------------------------------
    # plumbing
    # ------------------------------------------------------------------

    def _query_jsoneachrow(self, sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Run a query and parse JSONEachRow output into Python dicts."""
        raw = self._get_session().query(sql, "JSONEachRow", params=params or {})
        text = raw if isinstance(raw, str) else str(raw)
        rows: list[dict[str, Any]] = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))
        return rows


# Short alias for the common case. Documented in ``__init__.py`` re-exports.
ChDB = ChDBVectorStore
