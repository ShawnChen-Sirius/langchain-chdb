# langchain-chdb

[LangChain](https://github.com/langchain-ai/langchain) provider for [chDB](https://github.com/chdb-io/chdb) — the in-process OLAP SQL engine powered by ClickHouse.

`langchain-chdb` lets you use chDB as a vector store, document loader, chat-history store, and SQL backend for LangChain agents. Everything runs in the agent's own process; no server to operate. Federation to remote ClickHouse Cloud clusters is available through chDB's `remoteSecure()` table function.

> Status: 0.1.0a0 scaffolding. Public classes (`ChDBLoader`, `ChDBVectorStore`, `ChDBChatMessageHistory`) land incrementally per the v0.1 execution plan and are available in 0.1.0.

## What this gives you

- **Single engine for retrieval and analytics.** Native vector type plus `cosineDistance` / `L2Distance` / `dotProduct` so RAG and analytical SQL run in the same process.
- **Federation built in.** A LangChain agent running against a local Parquet file can `JOIN` it with a ClickHouse Cloud cluster via `remoteSecure()` in a single query.
- **1000+ ClickHouse functions.** `windowFunnel`, `uniqHLL12`, `geoToH3`, typed JSON, and the rest of the ClickHouse SQL surface are reachable from agent tools.

## Install

```bash
# Core: vector store, document loader, chat history
pip install langchain-chdb

# With SQLDatabaseToolkit support (pulls chdb-sqlalchemy)
pip install "langchain-chdb[sql]"
```

## Components

### `ChDBVectorStore`

```python
from langchain_chdb import ChDBVectorStore
from langchain_openai import OpenAIEmbeddings

store = ChDBVectorStore.from_texts(
    texts=["chDB is an embedded ClickHouse.", "It runs SQL on local files."],
    embedding=OpenAIEmbeddings(),
    embedding_dimension=1536,
    database="./chdb-store",
)

results = store.similarity_search("which engine embeds ClickHouse?", k=1)
```

Backed by `Array(Float32)` columns and the ClickHouse vector-distance functions. The constructor short alias `ChDB = ChDBVectorStore` is exported for brevity.

### `ChDBLoader`

```python
from langchain_chdb import ChDBLoader

loader = ChDBLoader(
    query="SELECT title, body FROM file('articles.parquet')",
    page_content_columns=["body"],
    metadata_columns=["title"],
)
docs = loader.load()
```

A single-column `page_content_columns` returns the raw cell as `Document.page_content`; multi-column or `None` serializes the listed columns as `col: value` lines. See [`docs/decisions/loader_page_content_format.md`](docs/decisions/loader_page_content_format.md) for the rationale.

### `ChDBChatMessageHistory`

```python
from langchain_chdb import ChDBChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

history = ChDBChatMessageHistory(session_id="abc", database="./chats.chdb")
history.add_message(HumanMessage("Hello"))
history.add_message(AIMessage("Hi!"))

for m in history.messages:
    print(type(m).__name__, m.content)
```

Implements `BaseChatMessageHistory`. Sessions are strictly isolated by `session_id`. Compose with `ChDBVectorStore.as_retriever()` and `RunnableWithMessageHistory` for retrieval-augmented chat.

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

The chdb-sqlalchemy dialect handles reflection, type mapping, and the introspection contract that `SQLDatabaseToolkit` depends on.

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
| Repo scaffold, CI, publish workflow | shipped in 0.1.0a0 |
| `ChDBLoader` | landing in 0.1.0 |
| `ChDBVectorStore` | landing in 0.1.0 (LangChain `VectorStoreIntegrationTests` is the gating contract) |
| `ChDBChatMessageHistory` | landing in 0.1.0 |
| SQLDatabaseToolkit cookbook | landing in 0.1.0 |
| ClickHouse vector-similarity ANN indexes | planned for 0.2.x |
| `BaseMemory` adapter | not planned — composing `ChDBVectorStore.as_retriever()` + `ChDBChatMessageHistory` with `RunnableWithMessageHistory` is the recommended pattern in LangChain 1.x |

Detailed plan: see [docs/langchain_chdb_v0_1_plan.md](docs/langchain_chdb_v0_1_plan.md) once it lands inside the repo.

## License

Apache 2.0 — see [LICENSE](LICENSE).

## Related

- Main chDB repository: https://github.com/chdb-io/chdb
- chDB documentation: https://clickhouse.com/docs/chdb
- LLM-friendly index: https://clickhouse.com/docs/chdb/llms.txt
- SQLAlchemy dialect: https://github.com/chdb-io/chdb-sqlalchemy
- LangChain: https://github.com/langchain-ai/langchain
- Community: https://discord.gg/D2Daa2fM5K
