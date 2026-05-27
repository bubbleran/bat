# Releasing `bat-cli` to PyPI

The package is **`bat-cli`**; the installed command is **`bat`**. End users install with
`pipx install bat-cli` (or `uv tool install bat-cli`).

## TL;DR

```bash
cd cli
# 1. bump version = "X.Y.Z" in pyproject.toml (PyPI rejects re-uploading an existing version)
rm -rf dist/
uv build
unzip -l dist/bat_cli-*.whl | grep -E "cli.py|image_defaults.py"   # sanity check
export UV_PUBLISH_TOKEN=pypi-...your-token...
uv publish
```

## Steps in detail

### 0. Bump the version
PyPI never lets you re-publish a version that already exists. Bump
`version` in [`pyproject.toml`](pyproject.toml) before every release.

### 1. Build
```bash
cd cli
rm -rf dist/        # otherwise `uv publish` also re-uploads stale artifacts
uv build            # -> dist/bat_cli-X.Y.Z-py3-none-any.whl + .tar.gz
```

### 2. Verify the wheel
`cli.py` and `image_defaults.py` are top-level modules (src-layout). They are only
included because of `py-modules = ["cli", "image_defaults"]` in `pyproject.toml`.
If that line is ever lost, the installed `bat` fails with
`ModuleNotFoundError: No module named 'cli'`. Guard against it:
```bash
unzip -l dist/bat_cli-*.whl | grep -E "cli.py|image_defaults.py"
```
Both files must appear at the top level.

### 3. Publish
```bash
export UV_PUBLISH_TOKEN=pypi-...your-token...    # pypi.org token
uv publish                                       # uploads everything in dist/
```
Never paste the token into chat, commit it, or put it in shell history. Use the env
var or the interactive prompt. If a token leaks, revoke it on pypi.org immediately.

## Optional: dry run on TestPyPI

TestPyPI is a **separate** site with its own account and its own token.
```bash
export UV_PUBLISH_TOKEN=pypi-...TESTPYPI-token...
uv publish --publish-url https://test.pypi.org/legacy/

# install back from TestPyPI (deps still come from real PyPI):
uv tool install --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  --prerelease=allow bat-cli
```

## Known issue: the `bat-adk` pre-release pin

`bat-cli` currently pins `bat-adk[openai]>=2026.06rc1`. The only `bat-adk` release that
satisfies this is the pre-release `2026.6rc1` (latest stable is `2026.3`). Because
pipx/uv exclude pre-releases by default, end users must install with a pre-release flag:

```bash
pipx install --pip-args="--pre" bat-cli
uv tool install --prerelease=allow bat-cli
```

**Fix when a stable `bat-adk` ships:** relax the pin to `bat-adk[openai]>=2026.6`
(or `>=2026.3` if compatible), bump `bat-cli`, and republish. Then `pipx install bat-cli`
works with no flags.
