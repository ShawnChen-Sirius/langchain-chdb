# Decision — `ChDBLoader.page_content` formatting

**Date:** 2026-05-21
**Status:** Adopted (v0.1)
**Recorded in plan:** [`langchain_chdb_v0_1_plan.md`](../../docs/langchain_chdb_v0_1_plan.md) D4
**Authority for this decision:** internal decision record (per Hard Constraint 1, this file may compare with alternative loaders for context; shipped code and the public README may not)

## Context

`ChDBLoader` returns one `Document` per row. Each `Document` has a
single `page_content: str` and a `metadata: dict`. There is no single
correct mapping from "row of N columns" to `(page_content, metadata)` —
different downstream uses need different shapes. The loader needs a
small, predictable rule set.

Two natural shapes:

* **Raw single-column value** — `page_content = str(row["body"])`. Best
  for the standard RAG ingestion path where exactly one column holds the
  text the embedding should cover.
* **Multi-column serialization** — `page_content = "title: ...\nbody: ..."`.
  Best when the document representation includes a few structured fields
  (title, body, source) that should all flow into the prompt and the
  embedding.

Several other LangChain loaders default to the multi-column serialization
shape because they don't know which column is the body. That works for
exploration but loses fidelity for the RAG path — concatenating `id: 1\n`
before the body shifts the embedding and pollutes retrieval.

## Decision

`ChDBLoader` uses the following rule, controlled by `page_content_columns`:

| `page_content_columns` argument | Behavior | Recommended use |
| --- | --- | --- |
| `None` (default) | Serialize every selected column as `col: value` lines | Exploratory / full-text-style ingestion when the user hasn't picked a body column |
| Single column, e.g. `["body"]` | Use `str(row["body"])` raw, no formatting | RAG main path — the embedding sees the body bytes only |
| Multi-column, e.g. `["title", "body"]` | Serialize the listed columns as `col: value` lines, in the given order | When several structured fields should be visible to the embedding |

`metadata_columns` is orthogonal:

* `None` (default): every column not in `page_content_columns` becomes
  metadata. If `page_content_columns` is also `None`, metadata is empty
  — everything went into `page_content`.
* Explicit list: only those columns become metadata.

## Rationale

1. **Single-column raw is the RAG default.** The most common
   `ChDBLoader` use is `loader.load()` on a parquet file with one body
   column. Returning `body: <text>\n` instead of `<text>` would break
   downstream cosine retrieval scores in measurable ways.

2. **Multi-column serialization is still available**, by listing the
   columns explicitly. The user opts in to that shape rather than
   getting it by default.

3. **Default `None` for both is safe** because it never silently drops
   data: every selected column lands somewhere in the `Document`.

## Non-goals

* Configurable separators, custom templates, JSON serialization, etc.
  Out of scope for v0.1. If a user needs a non-default shape, they
  control it via SQL (`SELECT concat(title, ' — ', body) AS combined`)
  or a post-processing step on the returned `Document` list.
* Auto-detecting a "body" column heuristically. Explicit is safer than
  guessing from column names.

## Alternatives considered

* **Always single-column when `page_content_columns` has length 1.**
  This is what we adopted.
* **Always serialize as `col: value`.** Rejected because it pollutes
  embeddings on the RAG main path.
* **Always return JSON.** Rejected because the embedding model would
  spend tokens parsing the JSON shape rather than the body text.

## Test coverage

`tests/test_document_loaders.py`:

* `test_single_column_returns_raw_content`
* `test_multi_column_uses_key_value_format`
* `test_default_concats_all_columns`
* `test_metadata_default_excludes_page_content_columns`
* `test_metadata_columns_explicit_when_page_content_columns_none`
