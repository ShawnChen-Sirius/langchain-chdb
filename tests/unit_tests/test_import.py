"""Smoke test that the package imports and exposes a version string.

Exists so ``pytest`` returns exit code 0 (instead of 5 — "no tests
collected") on the empty-but-scaffolded repo. Replaced with the real
behavioral tests as Steps 2-4 of the v0.1 plan land.
"""

from __future__ import annotations


def test_package_imports() -> None:
    import langchain_chdb

    assert langchain_chdb.__version__, "package must expose a non-empty __version__"


def test_version_is_pep440_pre_release() -> None:
    """The scaffold ships as 0.1.0a0; real releases bump per the v0.1 plan."""
    import re

    import langchain_chdb

    # PEP 440 release segments — accept N.N.N optionally followed by a pre-release tag.
    assert re.match(
        r"^\d+\.\d+\.\d+([abc]\d+|rc\d+|\.post\d+|\.dev\d+)?$",
        langchain_chdb.__version__,
    ), f"version {langchain_chdb.__version__!r} is not PEP 440 compliant"
