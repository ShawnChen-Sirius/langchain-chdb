# Release checklist

Run this before pushing a `v*` tag.

## Code health

```bash
uv pip install -e ".[sql,test]"
uv run pytest -q                                                  # full suite
uv run pytest -q tests/integration_tests/test_vectorstore_conformance.py
uv run ruff check .
uv run mypy langchain_chdb
```

All four must pass. The conformance suite is the gating event — never
ship a release with conformance red.

## Build sanity

```bash
rm -rf dist
uv build
uv run --with twine twine check dist/langchain_chdb-*
```

Verify:

- `dist/langchain_chdb-<VERSION>-py3-none-any.whl` exists.
- `dist/langchain_chdb-<VERSION>.tar.gz` exists.
- `twine check` reports `PASSED` for both.
- Run `unzip -p dist/langchain_chdb-*.whl langchain_chdb-*.dist-info/METADATA | head -30`:
  - `Version:` matches the intended tag (e.g. `0.1.0`).
  - `Author-email:` is `Shawn Chen <changshuo.chen@clickhouse.com>`.
  - `Requires-Python:` is `>=3.10`.

## Smoke run

```bash
uv run python scripts/docs_vectorstore_smoke.py
```

Expect the script to end with `ALL_OK`. The smoke script doesn't need
API keys or network access — it uses a local hash-based embedder.

## Version cross-check

```bash
grep -E '^version|^__version__' pyproject.toml langchain_chdb/__init__.py
```

Both must show the same version, matching the tag you are about to
push. The publish workflow refuses to upload if they disagree.

## CHANGELOG

- `CHANGELOG.md` has an `## [VERSION] — YYYY-MM-DD` section that
  matches the tag.
- Section body covers Added / Changed / Fixed / Deprecated as
  appropriate.
- Positioning paragraph reflects what this release means for
  LangChain agent workflows (not just the changed surface).

## Tag and push

```bash
git -c user.email=changshuo.chen@clickhouse.com -c user.name="Shawn Chen" \
    tag -a vVERSION -m "Release vVERSION"
git push upstream vVERSION    # triggers .github/workflows/publish.yml
```

The `-c` overrides are required if the local `git config user.email`
does not match the project committer identity; they apply for the
single `tag -a` invocation without mutating any git config file.

## Post-publish verification

In a clean venv:

```bash
uv run --with "langchain-chdb==VERSION" --no-project python -c "
from langchain_chdb import ChDBLoader, ChDBVectorStore, ChDB, ChDBChatMessageHistory, DistanceStrategy, __version__
assert __version__ == 'VERSION'
print('imports OK:', __version__)
"
```

Open `https://pypi.org/project/langchain-chdb/VERSION/` and confirm:

- Long description renders without broken Markdown.
- Classifiers reflect intended `Development Status`.
- License is Apache-2.0.
- Project URLs point at `chdb-io/langchain-chdb`, not a fork.

Open `https://github.com/chdb-io/langchain-chdb/releases/tag/vVERSION`
and confirm wheel + sdist are attached and downloadable.

## If something goes wrong

- **Publish workflow fails before upload**: fix locally, push a new tag
  with a bumped patch version. Do not retry the same tag — PyPI rejects
  duplicate filenames even if the previous upload technically failed.
- **Publish workflow fails after upload to PyPI**: the version is now
  permanent on PyPI. Ship a follow-up patch release with the fix.
  Document the failed version in `CHANGELOG.md` with a `Yanked:` line
  if appropriate.
- **GitHub Release missing assets**: re-run the workflow via
  `gh workflow run publish.yml --ref vVERSION --field tag=vVERSION`.
  The publish step is idempotent against existing PyPI files (it'll
  fail-fast with "File already exists") and `gh release upload
  --clobber` will replace any partial asset uploads.
