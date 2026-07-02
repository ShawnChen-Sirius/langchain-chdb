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


def test_quote_identifier_delegates_but_keeps_strict_policy():
    assert quote_identifier("events") == "`events`"
    with pytest.raises(ValueError, match="Invalid identifier"):
        quote_identifier("a; DROP TABLE t")
