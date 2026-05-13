# Contributing

This is a personal portfolio project, but the codebase is open for reference, study, and reuse. If you fork or build on it, the notes below will save you time.

## Project Philosophy

The goal of this project is to demonstrate end-to-end data science and ML engineering practices. Code is written to be readable first and clever second. Anything that would obscure intent for the sake of brevity has been left out.

## Local Setup

The simplest path is the local Python setup described in the main [README](README.md#option-2-local-python-no-docker). It uses SQLite, requires no Docker, and runs the entire pipeline in a few minutes on any laptop.

If you want to mirror the production-style stack, use `docker compose up --build` instead.

## Code Style

The codebase uses [Ruff](https://docs.astral.sh/ruff/) for linting. Configuration lives in `ruff.toml`. Run it with:

```bash
make lint
```

The CI pipeline runs the same check on every push, so anything that passes locally will pass remotely.

## Testing

The API has eighteen automated tests covering the health endpoint, single prediction, batch prediction, and input validation:

```bash
make test
```

When adding new endpoints or schema fields, add matching tests in `tests/test_api.py`.

## Adding New Features

A few conventions that help keep the project coherent:

- Configuration lives in `src/config.py`. New paths or constants belong there, not scattered across modules.
- Database access goes through `src/db.py`, which transparently handles both SQLite and PostgreSQL based on the `USE_POSTGRES` environment variable.
- Feature engineering should prefer SQL over pandas where the logic is naturally tabular (aggregations, joins, window functions). Python is used for transformations that are awkward in SQL.
- New plots from `src/models/evaluate.py` should be saved to `models/plots/`.

## Reporting Issues

If you find a bug or have a suggestion, please open a GitHub issue with a clear description and, where possible, a minimal reproduction.

## License

Contributions are accepted under the [MIT License](LICENSE).
