# Development

`rtnls_vascx` supports Python `3.10`, `3.11`, and `3.12`.

## Local Test Run

```bash
pip install -e ../rtnls_enface -e ../rtnls_fundusprep -e ".[test]"
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
