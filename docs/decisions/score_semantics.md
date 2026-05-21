# Decision — `ChDBVectorStore` score semantics

**Date:** 2026-05-22
**Status:** Adopted (v0.1)
**Affected component:** `langchain_chdb.vectorstores.ChDBVectorStore`
**Authority for this decision:** internal decision record

## Context

`ChDBVectorStore` exposes three similarity functions through
`DistanceStrategy`:

| Strategy | chDB function | Raw value direction |
| --- | --- | --- |
| `COSINE` | `cosineDistance(a, b)` | smaller is closer |
| `EUCLIDEAN` | `L2Distance(a, b)` | smaller is closer |
| `MAX_INNER_PRODUCT` | `dotProduct(a, b)` | **larger** is closer |

LangChain's `VectorStore` interface exposes two scoring methods:

1. `similarity_search_with_score` — returns `(Document, float)` pairs.
   No interface-level contract about whether the float is a distance,
   a similarity, or what range it lives in.
2. `similarity_search_with_relevance_scores` — returns
   `(Document, float)` pairs where the float must be a **relevance
   score in `[0, 1]`** with larger meaning more relevant.

Callers who want to apply a threshold (`score_threshold=0.5`) need a
predictable, comparable score, regardless of which `DistanceStrategy`
the store was constructed with.

## Decision

`similarity_search_with_score` returns **the raw chDB value**:

* `COSINE` → cosine distance (range `[0, 2]`, smaller is closer)
* `EUCLIDEAN` → L2 distance (range `[0, ∞)`, smaller is closer)
* `MAX_INNER_PRODUCT` → inner product (range `(-∞, ∞)`, larger is closer)

The docstring states the direction so callers who want to compare
scores within one strategy can do so. Mixing scores across strategies
is the caller's responsibility.

`similarity_search_with_relevance_scores` returns a **mapped `[0, 1]`
relevance**, monotone in semantic closeness for every strategy, using
a per-strategy formula:

| Strategy | Mapping |
| --- | --- |
| `COSINE` | `max(0, 1 - distance)` (clamps the upper tail since cosine distance can exceed 1 for non-unit-norm embeddings) |
| `EUCLIDEAN` | `1 / (1 + distance)` (smooth, asymptotic to 0 as distance grows) |
| `MAX_INNER_PRODUCT` | `sigmoid(score) = 1 / (1 + e^{-score})` (squashes the unbounded inner product into `(0, 1)`) |

`score_threshold` is applied to the relevance score after this
mapping, so a single threshold value (e.g. `0.5`) has the same "more
than halfway-relevant" semantic regardless of strategy.

## Rationale

1. **Two scoring methods, two purposes.** `similarity_search_with_score`
   is for callers who need the raw value (chained ranking, debugging,
   custom thresholds). `similarity_search_with_relevance_scores` is for
   LangChain idioms that assume a `[0, 1]` value.

2. **Monotone preservation matters.** Each mapping above preserves the
   order of the raw values within a strategy (closer raw → larger
   relevance). That keeps the LangChain idiom
   `[d for d, s in pairs if s >= score_threshold]` predictable.

3. **No cross-strategy comparability claimed.** A relevance of `0.7`
   under `COSINE` is not interchangeable with `0.7` under
   `MAX_INNER_PRODUCT`. Same-store same-strategy comparisons are
   well-defined; everything else is the caller's problem.

## Trade-offs

* The clamp on cosine relevance to `max(0, 1 - distance)` loses
  information when the raw distance exceeds 1 (which only happens for
  non-unit-norm embedding vectors). The clamp keeps the contract
  intact at the cost of compressing the worst-case tail.
* `sigmoid` for inner-product squashes large positive scores to ~1
  and large negative scores to ~0, which loses ranking resolution at
  the extremes. Within the typical magnitude range of dot products
  on unit-norm embeddings the resolution is fine.

## Non-goals

* Per-call override of the mapping function.
* User-supplied normalizers.

Both are open extensions but not part of v0.1.

## Test coverage

`tests/test_vectorstore.py`:

* `test_relevance_scores_in_unit_interval[cosine|euclidean|max_inner_product]`
  — every relevance score is in `[0, 1]` for every strategy.
* `test_relevance_score_monotone_with_raw_score` — relevance is
  non-increasing as raw distance increases.
* `test_relevance_mapping_cosine_clamped_to_zero` — cosine relevance
  clamps to 0 for raw distance > 1.
* `test_relevance_mapping_euclidean_is_inverse` — L2 mapping equals
  `1 / (1 + d)` at known anchor points.
* `test_relevance_mapping_inner_product_is_sigmoid` — inner-product
  mapping equals `sigmoid` at known anchor points.
* `test_score_threshold_filters_low_relevance_pairs` — `score_threshold`
  drops pairs whose mapped relevance falls below the threshold.
