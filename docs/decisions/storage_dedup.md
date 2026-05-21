# Decision — `ChDBVectorStore` storage-layer dedup strategy

**Date:** 2026-05-22
**Status:** Adopted (v0.1)
**Affected component:** `langchain_chdb.vectorstores.ChDBVectorStore`
**Authority for this decision:** internal decision record (per Hard Constraint 1, this file may compare alternatives for context; shipped code and README may not)

## Context

`ChDBVectorStore` upserts by `id`. LangChain's
`VectorStoreIntegrationTests` requires that
`add_documents([Document(id="1", page_content="x"), Document(id="1", page_content="y")])`
results in exactly one stored row with content `"y"`, and that
`delete(["1"])` followed immediately by `get_by_ids(["1"])` returns `[]`.

ClickHouse `MergeTree`'s `ORDER BY id` clause is a **sort key**, not a
**uniqueness constraint** — the table will gladly hold two rows with
the same `id` if both are inserted. A naive
`INSERT INTO vectors VALUES (id, ...)` would produce silent duplicates
that surface through `similarity_search` even if `get_by_ids` happens
to fold them.

## Decision

**v0.1 uses DELETE + INSERT with `mutations_sync = 1`** for upserts:

```sql
ALTER TABLE {table} DELETE WHERE id IN (...) SETTINGS mutations_sync = 1;
INSERT INTO {table} (id, content, metadata, embedding) VALUES (...);
```

The synchronous-mutation setting forces the delete to fully apply
before `INSERT` and any subsequent read. `add_documents` additionally
folds duplicate ids inside one input batch — keeping the last
occurrence — before the DELETE, so the upsert is well-defined even
when the caller passes `[Document(id="1"), Document(id="1")]`.

## Rationale

The contract test is the gating event for v0.1 shipping. DELETE +
INSERT with synchronous mutations guarantees read-after-write
correctness without any query-time dedup machinery. It costs one
mutation per write — fine at RAG ingest scales (1k–100k docs) and
documented as such.

## v0.2 plan: append-only / versioned dedup

The mutation cost makes the v0.1 protocol unsuitable for high-frequency
update workloads. v0.2 will likely move to:

* `ReplacingMergeTree(_ingest_ts)` engine with a hidden
  `_ingest_ts DateTime64(6)` column.
* Writes become pure appends, no mutations.
* Reads dedup at query time via `SELECT ... FROM {table} FINAL` or
  `argMax(payload, _ingest_ts) GROUP BY id`.
* Trade-off: query latency cost (FINAL forces merge-aware scan) for
  write throughput gain.
* Open question: whether to add a periodic `OPTIMIZE TABLE ... FINAL`
  schedule and how to expose it from the public API.

The v0.1 → v0.2 boundary is intentional. v0.1 is the
contract-passing version; v0.2 is the production-scale version. The
performance trade-off should not be solved inside v0.1 by relaxing
the synchronous mutation setting.

## Non-goals

* In-place updates (no equivalent of `UPDATE ... SET`). The
  delete-then-insert is the only mutation primitive on the v0.1 surface.
* Cross-process concurrency. Two writers against the same
  `id IN (...)` set in different processes can race the mutation. chDB
  itself does not guard against concurrent writers; out of scope for v0.1.

## Test coverage

`tests/test_vectorstore.py`:

* `test_repeated_id_in_same_batch_collapses_to_one_row` — same-batch
  duplicates fold to one physical row (asserts `len(get_by_ids) == 1`
  *and* `len(similarity_search) == 1`).
* `test_repeated_id_across_batches_overwrites` — second write wins.
* `test_delete_is_synchronously_visible` — read-after-delete sync.

The conformance suite (`tests/test_vectorstore_conformance.py`) covers
the wider LangChain contract, including
`test_add_documents_with_ids_is_idempotent`.
