"""Shared SQL-safety helpers for langchain-chdb.

Values are passed to chDB as server-side bound parameters (``{name:Type}`` +
``params=``), never string-interpolated — so there is no string-escaping helper
here. Identifiers cannot be bound, so they go through ``quote_identifier``, which
keeps langchain-chdb's strict "simple ASCII names only" policy and delegates the
actual quoting to ``chdb.agents.safety.quote_ident`` (the shared primitive that
``ChDBTool`` and mcp-clickhouse also use, so all chdb-io surfaces quote the same
way).
"""

import re

from chdb.agents.safety import quote_ident

__all__ = ["quote_identifier"]

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def quote_identifier(name: str) -> str:
    """Backtick-quote a simple SQL identifier, rejecting anything else loudly.

    langchain-chdb's public surface only accepts ``[A-Za-z_][A-Za-z0-9_]*``; the
    backtick quoting itself is done by the shared ``chdb.agents`` primitive.
    """
    if not isinstance(name, str) or not _IDENTIFIER_RE.match(name):
        raise ValueError(
            f"Invalid identifier {name!r}: must match [A-Za-z_][A-Za-z0-9_]*"
        )
    return quote_ident(name)
