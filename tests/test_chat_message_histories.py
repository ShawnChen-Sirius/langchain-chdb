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
* Reopening an instance against an existing on-disk database reads back
  prior writes via the public API alone.
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
