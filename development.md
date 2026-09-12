# Development

`rtnls_vascx` supports Python `3.10`, `3.11`, and `3.12`.

## Local Test Run

```bash
pip install -e ".[test]"
pytest
```

Useful variants:

```bash
pytest -m reference
pytest -m plotting
pytest -m profile
pytest --accept-vascx-reference
```

## Tox

Run the full tox matrix:

```bash
tox
```

Run a single environment:

```bash
tox -e py312
tox -e pkg
```

The committed `samples/fundus` and `tests/reference` fixtures are intentionally included so the tests work from a clean checkout and from the source release.

## Published-package regression checks

`tox -e public` runs the same inference and biomarker CLI regression tests against
PyPI packages, with no editable installations. Set `VASCX_VERSION` to select a
release, for example `VASCX_VERSION=1.2.3 tox -r -e public` (replace the example
version with an existing release). Use `tox -e public -- -m reference` for only
biomarker regression. See the README Testing section for inputs and tolerances.

The ordinary tox environments, including `pkg`, still use the local editable
package. Public testing uses the tests and references from this checkout; use a
matching source tag when checking an older release with a different interface.
