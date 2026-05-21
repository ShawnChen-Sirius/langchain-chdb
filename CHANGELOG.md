# Changelog

All notable changes to `langchain-chdb` are recorded here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-05-22

First public release. The `langchain-chdb` package brings chDB — the in-process
OLAP SQL engine powered by ClickHouse — to LangChain agents as a vector store,
document loader, chat-message history store, and SQL backend.

### Added

- **`ChDBLoader`** — `BaseLoader` that turns any chDB SQL query into LangChain
  `Document` objects. Common shapes:

  - `SELECT id, body FROM file('docs.parquet', 'Parquet')`
  - `SELECT * FROM s3('s3://bucket/key.parquet')`
  - `SELECT * FROM remoteSecure('host:9440', 'db.tbl', ...)`

  Result rows are parsed via chDB's `JSONEachRow` output format so `Array(T)`,
  `Map(K,V)`, and `JSON` cells arrive as native Python `list` / `dict` rather
  than `repr()`-style strings. `page_content_columns` controls how rows
  serialize into `Document.page_content`; unknown column names raise
  `ValueError` on the first row. See
  [`docs/decisions/loader_page_content_format.md`](docs/decisions/loader_page_content_format.md)
  for the formatting rationale.

- **`ChDBVectorStore`** (with `ChDB` short alias) — `VectorStore` backed by an
  `Array(Float32)` column with a `length(embedding) = N` `CHECK` constraint,
  stored in a `MergeTree` table sorted by `id` (sort key, not a uniqueness
  constraint — see the storage-dedup decision record). Passes the full
  `langchain_tests.integration_tests.vectorstores.VectorStoreIntegrationTests`
  conformance suite. Features:

  - Three `DistanceStrategy` values: `COSINE` (`cosineDistance`),
    `EUCLIDEAN` (`L2Distance`), `MAX_INNER_PRODUCT` (`dotProduct`).
    `similarity_search_with_score` returns the raw chDB value;
    `similarity_search_with_relevance_scores` returns a strategy-specific
    `[0, 1]` relevance with optional `score_threshold` filtering. See
    [`docs/decisions/score_semantics.md`](docs/decisions/score_semantics.md).
  - **Idempotent upsert** via `DELETE WHERE id IN (...) SETTINGS mutations_sync = 1`
    followed by `INSERT`, with same-batch id folding (last write wins). See
    [`docs/decisions/storage_dedup.md`](docs/decisions/storage_dedup.md) for
    why v0.1 uses synchronous mutations and what v0.2 will change.
  - **Whitelist metadata-filter DSL** — `$in` / `$gt` / `$gte` / `$lt` /
    `$lte` / `$ne` / `$and` / `$or` / `$not`. Unknown operators raise
    `ValueError` before any SQL is emitted; identifiers must match
    `[A-Za-z_][A-Za-z0-9_]*`.
  - `Document.id` is **never mutated** on the caller's input.
  - On-disk reopen reads back prior writes through the public API alone —
    `system.tables` probe on first read.
  - Full sync + async parity (`aadd_*`, `aget_by_ids`, `asimilarity_*`,
    `adelete`, `afrom_*`).

- **`ChDBChatMessageHistory`** — `BaseChatMessageHistory` backed by a
  `MergeTree` keyed on `(session_id, seq)`. `seq UInt64` is the canonical
  insertion-order key — assigned per session as `max(seq) + 1` at write time
  so the order survives wall-clock movement (NTP corrections, manual
  adjustments, DST rollover). `payload JSON` stores the full
  `message_to_dict()` output, so `HumanMessage` / `AIMessage` /
  `SystemMessage` / `ToolMessage` round-trip with type, content,
  `additional_kwargs`, and `tool_call_id` preserved. Sessions are strictly
  isolated — every read, write, and `clear()` is scoped to one `session_id`.
  Full sync + async parity (`aadd_messages`, `aget_messages`, `aclear`).

  The `max(seq) + 1` write protocol assumes a **single writer per
  session_id**. Two threads in the same Python process — or two separate
  processes against the same on-disk database — can race the `max(seq)`
  read and produce duplicate `seq` values. Multi-writer safety is out of
  scope for v0.1.

  *Not* a `BaseMemory` subclass: classic memory is deprecated in
  LangChain 1.x. Compose `ChDBVectorStore.as_retriever()` with
  `RunnableWithMessageHistory(ChDBChatMessageHistory)` for RAG-augmented
  chat.

- **SQLDatabaseToolkit path** — pulls `chdb-sqlalchemy >= 0.2.1` under the
  `[sql]` extra. A worked end-to-end example with LangGraph + Claude ships
  at [`docs/cookbook/text_to_sql_with_langgraph.ipynb`](docs/cookbook/text_to_sql_with_langgraph.ipynb).

- **CI infrastructure** — Python 3.10–3.13 × Ubuntu / macOS matrix, ruff +
  mypy + pytest. A "no comparative language" grep gate keeps shipped
  artifacts (source, README, CHANGELOG, public docs) from referring to
  alternative libraries by name; internal decision records under
  `docs/decisions/` are exempt.

- **Tag-triggered publish workflow** — `git tag v0.1.0 && git push upstream
  v0.1.0` builds wheel + sdist, runs `twine check`, uploads to PyPI via
  `PYPI_API_TOKEN`, and creates/uploads to the GitHub Release. A pre-build
  cross-check refuses to publish if the tag version disagrees with
  `pyproject.toml`.
