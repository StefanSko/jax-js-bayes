# Reference draws for posteriordb models

This directory contains manually generated reference draws for posteriordb models
that do not ship reference draws or summary stats upstream.

## Format

Reference draws are stored in **exact posteriordb format**:

- `{posterior_name}.json.zip` - Compressed JSON with draws (list of chain dicts)
- `{posterior_name}.info.json` - Metadata, diagnostics, and quality checks

This makes it easy to contribute these reference draws back to posteriordb.

## Generated reference files (current)

- `radon_all-radon_pooled`
- `radon_mn-radon_hierarchical_intercept_noncentered`
- `wells_data-wells_dist`

## Generating Reference Draws

To generate reference draws for any posteriordb model:

```bash
uv run python tests/posteriordb/reference_draws/generate_reference.py {posterior_name}

# Example:
uv run python tests/posteriordb/reference_draws/generate_reference.py wells_data-wells_dist
```

The script uses posteriordb's standard settings (10 chains, 10k warmup, 10k
sampling, thin=10, seed=4711).

## Contributing to posteriordb

These reference draws are generated in exact posteriordb format. To contribute:

1. Fork https://github.com/stan-dev/posteriordb
2. Copy files to the appropriate directories:
   - `{name}.json.zip` → `posterior_database/reference_posteriors/draws/draws/`
   - `{name}.info.json` → `posterior_database/reference_posteriors/draws/info/`
3. Submit a pull request

The info.json includes all required metadata (inference settings, diagnostics,
quality checks, version info) that posteriordb expects.

## Requirements

- cmdstanpy (`uv add --dev cmdstanpy`)
- CmdStan installed (`python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"`)
- posteriordb package and local database

See `docs/POSTERIORDB_REFERENCE_DRAWS.md` for the full workflow.
