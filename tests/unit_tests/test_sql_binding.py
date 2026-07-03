"""Unit tests for the SQL-safety layer (shared quoting + parameter binding).

These pin that filter *values* are bound as chDB params (never interpolated) and
that identifier quoting goes through the shared chdb.agents primitive while
keeping langchain-chdb's strict "simple names only" policy.
"""

import pytest

from langchain_chdb._sql import quote_identifier
from langchain_chdb.vectorstores import _filter_to_sql


def test_filter_binds_string_values_not_inlined():
    params: dict = {}
    sql = _filter_to_sql({"category": "a'; DROP TABLE t;--"}, "metadata", params)
    # the value never appears in the SQL text — it is a bound parameter
    assert "DROP TABLE" not in sql
    assert "{p0:String}" in sql
    assert params == {"p0": "a'; DROP TABLE t;--"}


def test_filter_binds_operator_and_in_values():
    params: dict = {}
    sql = _filter_to_sql({"year": {"$gte": 2020}, "tag": {"$in": ["x", "y"]}}, "metadata", params)
    # numeric + list members are all bound, none inlined
    assert "2020" not in sql and "'x'" not in sql
    assert set(params.values()) == {2020, "x", "y"}


def test_none_equality_is_null_not_bound():
    # `{field: None}` must become `isNull(...)`, not `= NULL` (never true).
    params: dict = {}
    sql = _filter_to_sql({"deleted_at": None}, "metadata", params)
    assert "isNull(" in sql
    assert "NULL" not in sql.replace("isNull", "")  # no bare NULL literal
    assert params == {}


def test_ne_none_is_not_null():
    params: dict = {}
    sql = _filter_to_sql({"deleted_at": {"$ne": None}}, "metadata", params)
    assert "isNotNull(" in sql
    assert params == {}


def test_in_with_none_member_uses_is_null():
    params: dict = {}
    sql = _filter_to_sql({"tag": {"$in": ["x", None]}}, "metadata", params)
    assert "isNull(" in sql
    assert set(params.values()) == {"x"}


def test_ordering_operator_rejects_none():
    for op in ("$gt", "$gte", "$lt", "$lte"):
        with pytest.raises(ValueError, match="does not accept None"):
            _filter_to_sql({"year": {op: None}}, "metadata", {})


def test_quote_identifier_delegates_but_keeps_strict_policy():
    assert quote_identifier("events") == "`events`"
    with pytest.raises(ValueError, match="Invalid identifier"):
        quote_identifier("a; DROP TABLE t")
