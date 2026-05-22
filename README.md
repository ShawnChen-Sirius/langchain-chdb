# langchain-chdb

[LangChain](https://github.com/langchain-ai/langchain) provider for [chDB](https://github.com/chdb-io/chdb) — the in-process OLAP SQL engine powered by ClickHouse.

`langchain-chdb` lets you use chDB as a vector store, document loader, chat-history store, and SQL backend for LangChain agents. Everything runs in the agent's own process; no server to operate. Federation to remote ClickHouse Cloud clusters is available through chDB's `remoteSecure()` table function, so the same agent can work over local files, persisted local state, and warehouse-scale ClickHouse tables.

> Status: v0.1.0 — first public release. Public surface: `ChDBLoader`, `ChDBVectorStore` (with `ChDB` short alias and `DistanceStrategy`), `ChDBChatMessageHistory`. Installable via `pip install langchain-chdb`. PyPI classifier is `Development Status :: 3 - Alpha` while the surface is in early adoption; bumps to Beta / Stable will follow once the API is exercised against more LangChain agent patterns.

## What this gives you

- **Embedded ClickHouse for agents.** Run ClickHouse SQL inside the LangChain process for local notebooks, CI fixtures, edge jobs, and agent sandboxes, then keep the same SQL shape when moving to ClickHouse Server or ClickHouse Cloud.
- **Single engine for retrieval and analytics.** `ChDBVectorStore`, `ChDBLoader`, `ChDBChatMessageHistory`, and the SQLDatabaseToolkit path all sit on the same chDB engine, so RAG, chat state, structured filters, and analytical SQL can compose without a separate service.
- **SQL-shaped document loading.** Use chDB table functions such as `file()`, `s3()`, `url()`, and `remoteSecure()` to turn Parquet, CSV, JSON, S3 objects, URLs, and remote ClickHouse tables into LangChain `Document` objects with `SELECT`, `WHERE`, `JOIN`, `GROUP BY`, and `LIMIT` before embedding.
- **Agent event and JSON analytics.** Store tool-call payloads, trace events, session metadata, and retrieval metadata in ClickHouse-style tables and query them with typed JSON paths, `MergeTree` storage, and analytical aggregates.
- **Native vector distance functions.** Exact search uses `Array(Float32)` embeddings plus `cosineDistance` / `L2Distance` / `dotProduct`; ClickHouse vector-similarity ANN indexes are planned for the 0.2 series.
- **1000+ ClickHouse functions.** `windowFunnel`, `sequenceMatch`, `retention`, `quantilesTDigest`, `uniqHLL12`, `geoToH3`, and the rest of the ClickHouse SQL surface are reachable from agent tools.
- **Federation built in.** A LangChain agent running against a local Parquet file can `JOIN` it with a ClickHouse Cloud cluster via `remoteSecure()` in a single query.

## Good fits

- Text-to-SQL agents that should generate ClickHouse SQL locally before running against production ClickHouse.
- Retrieval-augmented analytics over logs, events, traces, tickets, documents, and structured metadata.
- Local RAG workflows that need both vector retrieval and SQL filters over the same persisted store.
- Notebook-to-production workflows where the local prototype, CI smoke test, and ClickHouse-backed deployment should share one SQL dialect.

## Install

```bash
# Core: vector store, document loader, chat history
pip install langchain-chdb

# With SQLDatabaseToolkit support (pulls chdb-sqlalchemy)
pip install "langchain-chdb[sql]"
```

## Available now

### `ChDBLoader`

```python
from langchain_chdb import ChDBLoader

loader = ChDBLoader(
    query="SELECT title, body FROM file('articles.parquet', 'Parquet')",
    page_content_columns=["body"],
    metadata_columns=["title"],
)
docs = loader.load()
```

A single-column `page_content_columns` returns the raw cell as `Document.page_content`; multi-column or `None` serializes the listed columns as `col: value` lines. Bad column names raise `ValueError` on first row. See [`docs/decisions/loader_page_content_format.md`](docs/decisions/loader_page_content_format.md) for the rationale.

### `ChDBVectorStore`

```python
from langchain_chdb import ChDBVectorStore, DistanceStrategy
from langchain_openai import OpenAIEmbeddings

store = ChDBVectorStore.from_texts(
    texts=["chDB is an embedded ClickHouse.", "It runs SQL on local files."],
    embedding=OpenAIEmbeddings(),
    embedding_dimension=1536,
    database="./chdb-store",
    distance_strategy=DistanceStrategy.COSINE,
)

results = store.similarity_search("which engine embeds ClickHouse?", k=1)
```

Backed by an `Array(Float32)` column with a `length(embedding) = N` `CHECK` constraint, stored in a `MergeTree` table sorted by `id` (sort key — not a uniqueness constraint, see [`docs/decisions/storage_dedup.md`](docs/decisions/storage_dedup.md)). Supports `DistanceStrategy.COSINE` / `EUCLIDEAN` / `MAX_INNER_PRODUCT`, a whitelist metadata-filter DSL (`$in`, `$gt`/`$gte`/`$lt`/`$lte`/`$ne`, `$and`/`$or`/`$not`), idempotent upsert via `DELETE WHERE id IN (...) SETTINGS mutations_sync = 1` then `INSERT`, and `score_threshold` filtering on relevance. Passes LangChain's full `VectorStoreIntegrationTests` conformance suite. The short alias `ChDB = ChDBVectorStore` is exported for brevity.

### `ChDBChatMessageHistory`

```python
from langchain_chdb import ChDBChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

history = ChDBChatMessageHistory(session_id="abc", database="./chats.chdb")
history.add_messages([HumanMessage("Hello"), AIMessage("Hi!")])

for m in history.messages:
    print(type(m).__name__, m.content)
```

Implements `BaseChatMessageHistory` with `(session_id, seq)`-ordered `MergeTree` storage. The `seq UInt64` column is the canonical insertion-order key — assigned per session as `max(seq) + 1` at write time, immune to wall-clock movement (NTP corrections, manual adjustments, DST rollover). Sessions are strictly isolated; every read, write, and `clear()` is scoped to one `session_id`. All four core message types (`HumanMessage` / `AIMessage` / `SystemMessage` / `ToolMessage`) round-trip with type and content preserved, plus type-specific fields like `ToolMessage.tool_call_id` and `additional_kwargs`. The recommended retrieval-augmented chat pattern in LangChain 1.x is to compose `ChDBVectorStore.as_retriever()` with `RunnableWithMessageHistory(ChDBChatMessageHistory)` rather than to wrap them in a `BaseMemory` subclass.

> **Concurrency note.** The `max(seq) + 1` write protocol assumes a single writer per `session_id`. Two threads in the same Python process — or two separate processes against the same on-disk database — can race the `max(seq)` read and produce duplicate `seq` values. chDB itself does not guard against concurrent writers; multi-writer safety is out of scope for v0.1.

### SQLDatabaseToolkit integration

chDB plugs into LangChain's `SQLDatabaseToolkit` through the [`chdb-sqlalchemy`](https://github.com/chdb-io/chdb-sqlalchemy) dialect, exposed under the `[sql]` extra:

```python
from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit

engine = create_engine("chdb:///./my.chdb")
db = SQLDatabase(engine)
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
```

The chdb-sqlalchemy dialect handles reflection, type mapping, and the introspection contract that `SQLDatabaseToolkit` depends on. A worked end-to-end example with LangGraph + Claude lives at [`docs/cookbook/text_to_sql_with_langgraph.ipynb`](docs/cookbook/text_to_sql_with_langgraph.ipynb).

## Reference architecture

```
LangChain agent
      │
      ▼
langchain-chdb  ◀── this package
      │
      ▼
chDB (in-process)
      │       ╲
      ▼        ╲
Parquet/CSV/   remoteSecure() ──► ClickHouse Cloud
S3/HTTP files
```

No external services beyond what the agent already uses (LLM API, optional remote ClickHouse cluster). Retrieval and analytical SQL happen inside the agent's process.

## Status

| Component | State |
|---|---|
| `ChDBLoader` | available on PyPI from 0.1.0 |
| `ChDBVectorStore` (and `ChDB` short alias) | available on PyPI from 0.1.0. Passes LangChain's `VectorStoreIntegrationTests`. |
| `ChDBChatMessageHistory` | available on PyPI from 0.1.0 |
| Text-to-SQL cookbook (LangGraph + Claude) | shipped in repo at [`docs/cookbook/`](docs/cookbook/); runnable with `ANTHROPIC_API_KEY` |
| **Exact vector search** via `cosineDistance` / `L2Distance` / `dotProduct` | shipped in 0.1.0 |
| ClickHouse vector-similarity ANN indexes | **not in 0.1.0** — planned for 0.2.x |
| Append-only / `ReplacingMergeTree` vector storage | planned for 0.2.x; see [`docs/decisions/storage_dedup.md`](docs/decisions/storage_dedup.md) |
| `BaseMemory` adapter | not planned — `ChDBVectorStore.as_retriever()` + `RunnableWithMessageHistory(ChDBChatMessageHistory)` is the recommended composition in LangChain 1.x |

## LangChain docs readiness

For maintainers building the `langchain-ai/langchain` integration docs PR:

- The LangChain `VectorStoreIntegrationTests` conformance suite passes in [`tests/integration_tests/test_vectorstore_conformance.py`](tests/integration_tests/test_vectorstore_conformance.py) against `langchain-tests >= 1.1.8, < 1.2`.
- [`scripts/docs_vectorstore_smoke.py`](scripts/docs_vectorstore_smoke.py) is a self-contained smoke test — local fake embedder, no API keys — exercising add / metadata filter / `similarity_search` / `similarity_search_with_score` / `delete` / persistent reopen. The code blocks in any official docs page can come straight from this script.
- The Text-to-SQL cookbook ([`docs/cookbook/text_to_sql_with_langgraph.ipynb`](docs/cookbook/text_to_sql_with_langgraph.ipynb)) is an extended example, not part of the official docs PR — LangChain's docs repo prefers `.mdx` text over notebooks for new integrations.

## Decision records

The trade-offs that shaped the v0.1 surface live under
[`docs/decisions/`](docs/decisions/):

- [`loader_page_content_format.md`](docs/decisions/loader_page_content_format.md) — how `ChDBLoader` builds `Document.page_content` and `Document.metadata` from a query result.
- [`storage_dedup.md`](docs/decisions/storage_dedup.md) — why `ChDBVectorStore` v0.1 upserts via `DELETE WHERE id IN (...) SETTINGS mutations_sync = 1` + `INSERT`, and the v0.2 plan to migrate to append-only / versioned dedup.
- [`score_semantics.md`](docs/decisions/score_semantics.md) — how the three `DistanceStrategy` values map raw chDB distances into the `[0, 1]` LangChain relevance interval.

## License

Apache 2.0 — see [LICENSE](LICENSE).

## Related

- Main chDB repository: https://github.com/chdb-io/chdb
- chDB documentation: https://clickhouse.com/docs/chdb
- LLM-friendly index: https://clickhouse.com/docs/chdb/llms.txt
- SQLAlchemy dialect: https://github.com/chdb-io/chdb-sqlalchemy
- LangChain: https://github.com/langchain-ai/langchain
- Community: https://discord.gg/D2Daa2fM5K
