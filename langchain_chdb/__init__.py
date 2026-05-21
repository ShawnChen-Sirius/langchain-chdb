"""LangChain provider for chDB.

This package exposes chDB — the in-process OLAP SQL engine powered by
ClickHouse — to LangChain through:

* ``ChDBVectorStore`` (alias ``ChDB``): native-vector store backed by
  ``Array(Float32)`` columns and ``cosineDistance`` / ``L2Distance`` /
  ``dotProduct`` similarity functions.
* ``ChDBLoader``: document loader for SQL queries against any chDB
  table function (files, S3, remote ClickHouse, etc.).
* ``ChDBChatMessageHistory``: persistent chat-history store keyed by
  ``session_id``.

The SQLDatabaseToolkit path is available via the ``[sql]`` extra, which
pulls in the ``chdb-sqlalchemy`` dialect.

Public classes are added incrementally as the v0.1 plan steps land.
"""

from __future__ import annotations

from langchain_chdb.chat_message_histories import ChDBChatMessageHistory
from langchain_chdb.document_loaders import ChDBLoader
from langchain_chdb.vectorstores import ChDB, ChDBVectorStore, DistanceStrategy

__version__ = "0.1.0"

__all__ = [
    "ChDB",
    "ChDBChatMessageHistory",
    "ChDBLoader",
    "ChDBVectorStore",
    "DistanceStrategy",
    "__version__",
]
