"""Tests for ``ChDBChatMessageHistory`` (Step 4 of the v0.1 execution plan).

Coverage:

* All four LangChain message types (``HumanMessage`` / ``AIMessage`` /
  ``SystemMessage`` / ``ToolMessage``) round-trip with their original
  class and content preserved.
* ``additional_kwargs`` and ``ToolMessage.tool_call_id`` survive the
  round-trip (JSON payload stores the full ``message_to_dict`` output).
* Different ``session_id`` values produce strictly isolated histories.
* ``clear()`` removes only the current session's messages.
* ``messages`` returns rows in insertion order.
* Async (``aadd_messages`` / ``aget_messages`` / ``aclear``) parity with sync.
* Insertion order is preserved even when all messages collide on the
  wall-clock ``ts`` (the ``seq`` tie-breaker recovers order).
* Reopening an instance against a shared ``:memory:`` reads back prior
  writes via the public API alone (in-process case).
* Reopening an instance against an on-disk database file reads back
  prior writes via the public API alone (file-backed case, exercised
  in a subprocess to dodge chDB's process-global ``EmbeddedServer``).
"""

from __future__ import annotations

import pytest
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from langchain_chdb import ChDBChatMessageHistory

# ---------------------------------------------------------------------------
# conftest needs to also clear chat_history table — extend the default
# autouse fixture by adding our own.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_chat_history_state():
    from chdb.session import Session

    session = Session(path=":memory:")
    try:
        session.query("DROP TABLE IF EXISTS langchain_chdb_chat_history")
    finally:
        session.close()


# ---------------------------------------------------------------------------
# message-type round-trip
# ---------------------------------------------------------------------------


def _sample_messages() -> list[BaseMessage]:
    return [
        HumanMessage("hello", additional_kwargs={"meta": "demo"}),
        AIMessage("hi there"),
        SystemMessage("be brief"),
        ToolMessage("computed result", tool_call_id="call_42"),
    ]


def test_message_types_round_trip():
    h = ChDBChatMessageHistory(session_id="s1")
    originals = _sample_messages()
    h.add_messages(originals)
    received = h.messages
    assert [type(m).__name__ for m in received] == [
        "HumanMessage", "AIMessage", "SystemMessage", "ToolMessage",
    ]
    for orig, recv in zip(originals, received, strict=True):
        assert orig.content == recv.content


def test_additional_kwargs_survive_round_trip():
    h = ChDBChatMessageHistory(session_id="s1")
    h.add_messages([HumanMessage("hi", additional_kwargs={"x": 1, "y": [2, 3]})])
    recv = h.messages[0]
    assert recv.additional_kwargs == {"x": 1, "y": [2, 3]}


def test_tool_call_id_survives_round_trip():
    h = ChDBChatMessageHistory(session_id="s1")
    h.add_messages([ToolMessage("result", tool_call_id="call_42")])
    recv = h.messages[0]
    assert isinstance(recv, ToolMessage)
    assert recv.tool_call_id == "call_42"


# ---------------------------------------------------------------------------
# session isolation
# ---------------------------------------------------------------------------


def test_session_isolation_strict():
    a = ChDBChatMessageHistory(session_id="alice")
    b = ChDBChatMessageHistory(session_id="bob")
    a.add_messages([HumanMessage("alice-secret")])
    b.add_messages([HumanMessage("bob-secret")])

    assert len(a.messages) == 1
    assert a.messages[0].content == "alice-secret"
    assert len(b.messages) == 1
    assert b.messages[0].content == "bob-secret"


def test_clear_only_clears_current_session():
    a = ChDBChatMessageHistory(session_id="alice")
    b = ChDBChatMessageHistory(session_id="bob")
    a.add_messages([HumanMessage("from-alice")])
    b.add_messages([HumanMessage("from-bob")])
    a.clear()
    assert a.messages == []
    assert len(b.messages) == 1
    assert b.messages[0].content == "from-bob"


# ---------------------------------------------------------------------------
# ordering
# ---------------------------------------------------------------------------


def test_messages_return_in_insertion_order():
    h = ChDBChatMessageHistory(session_id="s1")
    h.add_messages([HumanMessage("first")])
    h.add_messages([AIMessage("second")])
    h.add_messages([HumanMessage("third")])
    contents = [m.content for m in h.messages]
    assert contents == ["first", "second", "third"]


def test_batch_insertion_preserves_order():
    h = ChDBChatMessageHistory(session_id="s1")
    h.add_messages([HumanMessage(f"msg{i}") for i in range(5)])
    assert [m.content for m in h.messages] == [f"msg{i}" for i in range(5)]


# ---------------------------------------------------------------------------
# edge cases
# ---------------------------------------------------------------------------


def test_empty_history():
    h = ChDBChatMessageHistory(session_id="never_used")
    assert h.messages == []


def test_clear_on_empty_history_is_noop():
    h = ChDBChatMessageHistory(session_id="never_used")
    h.clear()  # should not raise
    assert h.messages == []


def test_add_empty_list_is_noop():
    h = ChDBChatMessageHistory(session_id="s1")
    h.add_messages([])
    assert h.messages == []


def test_add_non_message_raises():
    h = ChDBChatMessageHistory(session_id="s1")
    with pytest.raises(TypeError, match=r"BaseMessage"):
        h.add_messages(["not a message"])  # type: ignore[list-item]


def test_invalid_table_name_rejected():
    with pytest.raises(ValueError, match=r"Invalid identifier"):
        ChDBChatMessageHistory(session_id="s1", table_name="bad-name")


def test_empty_session_id_rejected():
    with pytest.raises(ValueError, match=r"session_id"):
        ChDBChatMessageHistory(session_id="")


# ---------------------------------------------------------------------------
# async parity
# ---------------------------------------------------------------------------


async def test_aadd_aget_aclear_parity():
    h = ChDBChatMessageHistory(session_id="s1")
    await h.aadd_messages([HumanMessage("hi"), AIMessage("hello")])
    msgs = await h.aget_messages()
    assert [type(m).__name__ for m in msgs] == ["HumanMessage", "AIMessage"]
    await h.aclear()
    assert await h.aget_messages() == []


async def test_async_session_isolation():
    a = ChDBChatMessageHistory(session_id="alice")
    b = ChDBChatMessageHistory(session_id="bob")
    await a.aadd_messages([HumanMessage("a-msg")])
    await b.aadd_messages([HumanMessage("b-msg")])
    a_msgs = await a.aget_messages()
    b_msgs = await b.aget_messages()
    assert {m.content for m in a_msgs} == {"a-msg"}
    assert {m.content for m in b_msgs} == {"b-msg"}


# ---------------------------------------------------------------------------
# read-after-reopen through public API only
# ---------------------------------------------------------------------------


def test_reopen_via_public_api_sees_prior_writes():
    """A fresh ChDBChatMessageHistory against the shared ``:memory:``
    state must see existing messages without any private method calls."""
    s1 = ChDBChatMessageHistory(session_id="persistent")
    s1.add_messages([HumanMessage("a"), AIMessage("b")])

    s2 = ChDBChatMessageHistory(session_id="persistent")
    msgs = s2.messages
    assert [m.content for m in msgs] == ["a", "b"]


# ---------------------------------------------------------------------------
# tie-break ordering — same ts, different seq
# ---------------------------------------------------------------------------


def test_insertion_order_preserved_when_all_ts_collide(monkeypatch):
    """Even when every ``add_messages`` call observes the same wall-clock
    ``time.time()``, the persistent ``seq`` counter must recover
    insertion order on read.

    Reproduces the reviewer's case: five sequential single-message
    writes against a clock frozen at one instant.
    """
    import langchain_chdb.chat_message_histories as cmh_module

    monkeypatch.setattr(cmh_module.time, "time", lambda: 1_700_000_000.0)

    h = ChDBChatMessageHistory(session_id="s_tie")
    for i in range(5):
        h.add_messages([HumanMessage(str(i))])

    contents = [m.content for m in h.messages]
    assert contents == ["0", "1", "2", "3", "4"], (
        f"insertion order lost under same-ts collision: {contents!r}"
    )


def test_clock_rewind_does_not_reorder_history(monkeypatch):
    """The ``seq`` column is the canonical insertion-order key — a
    clock rewind between writes (NTP correction, manual adjustment,
    DST regression) must NOT reorder previously-written messages.

    Reproduces the reviewer's case: ``first`` is written at one
    wall-clock instant, then the clock jumps backward, then
    ``second`` is written. The read order must still be
    ``['first', 'second']``.
    """
    import itertools as _it

    import langchain_chdb.chat_message_histories as cmh_module

    # Each call to time.time() returns a smaller value than the previous one.
    clocks = _it.chain(
        [1_700_000_000.0, 1_500_000_000.0],
        _it.repeat(1_500_000_000.0),
    )
    monkeypatch.setattr(cmh_module.time, "time", lambda: next(clocks))

    h = ChDBChatMessageHistory(session_id="rewind")
    h.add_messages([HumanMessage("first")])
    h.add_messages([HumanMessage("second")])

    contents = [m.content for m in h.messages]
    assert contents == ["first", "second"], (
        f"clock rewind reordered the history: {contents!r}"
    )


def test_clear_then_add_resets_seq_correctly():
    """After ``clear()``, the session's stored rows are gone, so
    ``max(seq)`` becomes 0 (empty result) and the next write starts
    at seq=1 again. Insertion order in the new conversation must be
    correct."""
    h = ChDBChatMessageHistory(session_id="reset")
    h.add_messages([HumanMessage("old-1"), HumanMessage("old-2")])
    h.clear()
    assert h.messages == []
    h.add_messages([HumanMessage("new-1"), HumanMessage("new-2")])
    contents = [m.content for m in h.messages]
    assert contents == ["new-1", "new-2"]


# ---------------------------------------------------------------------------
# file-backed reopen — subprocess so the chDB process-global
# EmbeddedServer path can be bound cleanly
# ---------------------------------------------------------------------------


def test_file_backed_reopen_round_trip(tmp_path):
    """Write to a file-backed history, close, reopen the same path in
    a fresh process, read the messages back. chDB binds its
    ``EmbeddedServer`` once per process to whichever path is opened
    first — other tests in this file use ``:memory:``, so the on-disk
    round-trip has to run inside a subprocess to get a clean init.
    """
    import subprocess
    import sys
    import textwrap

    db = tmp_path / "chat.chdb"
    code = textwrap.dedent(f"""
        from langchain_core.messages import HumanMessage, AIMessage
        from langchain_chdb import ChDBChatMessageHistory

        # Phase 1: write two messages, close.
        s1 = ChDBChatMessageHistory(session_id='persist', database={str(db)!r})
        s1.add_messages([HumanMessage('disk-a'), AIMessage('disk-b')])
        s1.close()

        # Phase 2: fresh instance, SAME path, public API only.
        s2 = ChDBChatMessageHistory(session_id='persist', database={str(db)!r})
        msgs = s2.messages
        assert len(msgs) == 2, f"expected 2 messages, got {{len(msgs)}}"
        contents = [m.content for m in msgs]
        assert contents == ['disk-a', 'disk-b'], f"got {{contents!r}}"
        types = [type(m).__name__ for m in msgs]
        assert types == ['HumanMessage', 'AIMessage'], f"got {{types!r}}"
        s2.close()
        print('DISK_REOPEN_OK')
    """)
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"subprocess failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "DISK_REOPEN_OK" in result.stdout
