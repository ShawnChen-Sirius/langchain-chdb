"""ChDBChatMessageHistory — persistent chat-message storage in chDB.

Storage model
-------------

Each store maps to one chDB table::

    CREATE TABLE {table_name} (
        session_id String,
        seq        UInt64,
        ts         DateTime64(6),
        role       LowCardinality(String),
        payload    JSON
    )
    ENGINE = MergeTree()
    ORDER BY (session_id, seq);

* ``session_id`` partitions histories by conversation. Reads, writes,
  and ``clear()`` are all scoped to one ``session_id``; the schema
  enforces that by ordering the table on ``(session_id, seq)`` so
  a per-session lookup is a contiguous range scan.
* ``seq UInt64`` is the **canonical insertion-order key**, scoped per
  session. Each ``add_messages`` call queries the current
  ``max(seq)`` for the session, then assigns ``max + 1, max + 2, …``
  to its batch. Subsequent reads ``ORDER BY seq ASC``. The order is
  immune to wall-clock movement — NTP corrections, manual clock
  adjustments, daylight-saving rollover, etc. cannot reorder a
  history.
* ``ts DateTime64(6)`` is the wall-clock timestamp at write time. It
  is **not** used for ordering; it is informational, suitable for
  human-facing display or future time-range filtering.
* ``role`` is the LangChain ``BaseMessage.type`` (``human`` / ``ai`` /
  ``system`` / ``tool`` / ``chat`` / ``function``). Denormalized from
  the payload so SQL filtering and aggregation are cheap.
* ``payload JSON`` is the full ``message_to_dict()`` output. Reads
  reconstruct the original ``BaseMessage`` subclass via
  ``messages_from_dict``, which preserves type-specific fields like
  ``ToolMessage.tool_call_id`` and ``AIMessage.tool_calls``.

The per-session ``max(seq) + 1`` write protocol assumes a single
writer per session. Two writers against the same session — either
two threads in the same Python process or two separate processes
against the same on-disk database — can race the ``max(seq)`` read
and produce duplicate ``seq`` values. chDB itself does not guard
against concurrent writers either, and multi-writer correctness is
out of scope for v0.1.

The recommended retrieval-augmented chat pattern in LangChain 1.x is to
compose ``ChDBVectorStore.as_retriever()`` with
``RunnableWithMessageHistory(ChDBChatMessageHistory)`` rather than to
wrap them in a ``BaseMemory`` subclass — ``BaseMemory`` is deprecated.

Result-path note
----------------

Reads go through ``Session.query(sql, "JSONEachRow")`` so the
``payload`` JSON cell arrives as a native Python ``dict`` from
``json.loads`` rather than the ``repr()``-style string that the
``chdb.dbapi`` cursor would return.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
import time
from typing import Any

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _quote_identifier(name: str) -> str:
    if not isinstance(name, str) or not _IDENTIFIER_RE.match(name):
        raise ValueError(
            f"Invalid identifier {name!r}: must match [A-Za-z_][A-Za-z0-9_]*"
        )
    return f"`{name}`"


def _escape_string_literal(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"string expected, got {type(value).__name__}")
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"


def _format_json_literal(value: Any) -> str:
    if value is None:
        value = {}
    return _escape_string_literal(
        json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    )


class ChDBChatMessageHistory(BaseChatMessageHistory):
    """LangChain ``BaseChatMessageHistory`` backed by chDB.

    Insertion order is preserved via a persistent per-session ``seq``
    counter (see module docstring). Reads ``ORDER BY seq ASC``.

    Parameters
    ----------
    session_id:
        Conversation key. Every read / write / clear scoped to this id.
        Different session ids are strictly isolated — no method on the
        history exposes other sessions' rows.
    database:
        Path to a persistent chDB store, or ``":memory:"`` for a
        transient session. The history opens its own
        ``chdb.session.Session`` and keeps it alive for the instance's
        lifetime.
    table_name:
        Backing table name; defaults to ``langchain_chdb_chat_history``.
        Must match ``[A-Za-z_][A-Za-z0-9_]*``.
    create_if_not_exists:
        When ``True`` (default), the table is created on first read or
        write if it doesn't exist. ``False`` is the right setting when
        the schema is managed externally.
    """

    def __init__(
        self,
        session_id: str,
        *,
        database: str = ":memory:",
        table_name: str = "langchain_chdb_chat_history",
        create_if_not_exists: bool = True,
    ) -> None:
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be a non-empty string")
        _quote_identifier(table_name)

        self._session_id = session_id
        self._database = database
        self._table_name = table_name
        self._create_if_not_exists = create_if_not_exists

        self._session: Any = None
        self._table_initialized = False

    # ------------------------------------------------------------------
    # session / lifecycle
    # ------------------------------------------------------------------

    @property
    def session_id(self) -> str:
        return self._session_id

    def _get_session(self) -> Any:
        if self._session is None:
            from chdb.session import Session
            self._session = Session(path=self._database)
        return self._session

    def close(self) -> None:
        if self._session is not None:
            self._session.close()
            self._session = None
            self._table_initialized = False

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.close()

    # ------------------------------------------------------------------
    # DDL / table probing
    # ------------------------------------------------------------------

    def _table_exists_in_chdb(self) -> bool:
        rows = self._query_jsoneachrow(
            "SELECT 1 FROM system.tables "
            "WHERE database = currentDatabase() "
            f"AND name = {_escape_string_literal(self._table_name)} LIMIT 1"
        )
        return bool(rows)

    def _ready_for_read(self) -> bool:
        if self._table_initialized:
            return True
        if self._table_exists_in_chdb():
            self._table_initialized = True
            return True
        return False

    def _ensure_table(self) -> None:
        if self._table_initialized:
            return
        if not self._create_if_not_exists:
            self._table_initialized = True
            return
        ddl = (
            f"CREATE TABLE IF NOT EXISTS {_quote_identifier(self._table_name)} (\n"
            f"    session_id String,\n"
            f"    seq UInt64,\n"
            f"    ts DateTime64(6),\n"
            f"    role LowCardinality(String),\n"
            f"    payload JSON\n"
            f") ENGINE = MergeTree() ORDER BY (session_id, seq)"
        )
        self._get_session().query(ddl)
        self._table_initialized = True

    # ------------------------------------------------------------------
    # Sync interface required by BaseChatMessageHistory
    # ------------------------------------------------------------------

    def add_messages(self, messages: list[BaseMessage]) -> None:
        """Append messages to the current session.

        Each batch reads the session's current ``max(seq)`` and assigns
        consecutive ``max + 1, max + 2, …`` values to the new rows.
        Reads ``ORDER BY seq ASC``, so insertion order is recovered
        regardless of wall-clock movement.
        """
        if not messages:
            return

        # Validate types before opening a chDB session — bad input must
        # not leave a half-created table.
        for msg in messages:
            if not isinstance(msg, BaseMessage):
                raise TypeError(
                    f"Expected BaseMessage, got {type(msg).__name__}: {msg!r}"
                )

        self._ensure_table()
        next_seq = self._next_seq_for_session()

        rows: list[str] = []
        now_micro = int(time.time() * 1_000_000)
        for i, msg in enumerate(messages):
            ts_micro = now_micro + i  # informational; not used for ordering
            seq = next_seq + i
            payload = message_to_dict(msg)
            rows.append(
                "("
                f"{_escape_string_literal(self._session_id)}, "
                f"{seq}, "
                f"fromUnixTimestamp64Micro({ts_micro}), "
                f"{_escape_string_literal(msg.type)}, "
                f"{_format_json_literal(payload)}"
                ")"
            )

        self._get_session().query(
            f"INSERT INTO {_quote_identifier(self._table_name)} "
            "(session_id, seq, ts, role, payload) VALUES "
            + ",\n".join(rows)
        )

    def _next_seq_for_session(self) -> int:
        """Return ``max(seq) + 1`` for the current session, or ``1`` if empty.

        The query lives behind the lazy table-init path so it only runs
        after ``_ensure_table`` has guaranteed the table exists.
        """
        rows = self._query_jsoneachrow(
            f"SELECT max(seq) AS m FROM {_quote_identifier(self._table_name)} "
            f"WHERE session_id = {_escape_string_literal(self._session_id)}"
        )
        if not rows:
            return 1
        m = rows[0].get("m")
        # UInt64 max() over an empty set returns 0 in chDB; treat None
        # defensively the same way for any future Nullable migration.
        if m is None:
            return 1
        return int(m) + 1

    def clear(self) -> None:
        """Remove every message belonging to the current session.

        Other sessions in the same backing table are untouched. The
        delete is synchronous (``mutations_sync = 1``), so a subsequent
        ``messages`` read sees the empty history immediately.
        """
        if not self._ready_for_read():
            return
        self._get_session().query(
            f"ALTER TABLE {_quote_identifier(self._table_name)} "
            f"DELETE WHERE session_id = {_escape_string_literal(self._session_id)} "
            "SETTINGS mutations_sync = 1"
        )

    @property
    def messages(self) -> list[BaseMessage]:
        """Return all messages for this session in insertion order."""
        if not self._ready_for_read():
            return []
        rows = self._query_jsoneachrow(
            f"SELECT payload FROM {_quote_identifier(self._table_name)} "
            f"WHERE session_id = {_escape_string_literal(self._session_id)} "
            "ORDER BY seq ASC"
        )
        if not rows:
            return []
        return messages_from_dict([row["payload"] for row in rows])

    # ------------------------------------------------------------------
    # Async surface
    # ------------------------------------------------------------------
    #
    # chDB is sync; dispatch each call to a worker thread so the event
    # loop is not blocked. Behavior is identical to the sync path.

    async def aadd_messages(self, messages: list[BaseMessage]) -> None:
        await asyncio.to_thread(self.add_messages, messages)

    async def aclear(self) -> None:
        await asyncio.to_thread(self.clear)

    async def aget_messages(self) -> list[BaseMessage]:
        return await asyncio.to_thread(lambda: self.messages)

    # ------------------------------------------------------------------
    # plumbing
    # ------------------------------------------------------------------

    def _query_jsoneachrow(self, sql: str) -> list[dict[str, Any]]:
        raw = self._get_session().query(sql, "JSONEachRow")
        text = raw if isinstance(raw, str) else str(raw)
        out: list[dict[str, Any]] = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            out.append(json.loads(stripped))
        return out
