# Reference draws for posteriordb

This repo uses posteriordb data for integration tests. Some posteriors do not
ship reference draws or mean summaries upstream, so we keep local reference
zip files in `tests/posteriordb/reference_draws/` and compute mean stats from
those draws during tests.

## Prereqs

- Local posteriordb checkout. Either:
  - default path: `~/.posteriordb/posterior_database`, or
  - set `POSTERIORDB_PATH` to your checkout root.
- CmdStan and cmdstanpy installed (for sampling).
- Python environment with `posteriordb`, `cmdstanpy`, `numpy`.
- `uv` is recommended for reproducible runs.

## Generate reference draws

We keep a generator script at `tests/posteriordb/reference_draws/generate_reference.py`.
It mirrors posteriordb's reference draw format and uses standard settings:

- chains: 10
- iter_sampling: 10000
- iter_warmup: 10000
- thin: 10 (1000 draws/chain, 10000 total)
- seed: 4711

Run from the repo root:

```bash
uv run python tests/posteriordb/reference_draws/generate_reference.py \
  <posterior_name>
```

Examples:

```bash
uv run python tests/posteriordb/reference_draws/generate_reference.py \
  radon_all-radon_pooled
```

This writes:

- `tests/posteriordb/reference_draws/<posterior_name>.json.zip`
- `tests/posteriordb/reference_draws/<posterior_name>.info.json`

## How tests consume these

`tests/posteriordb/pdb.ts` checks `tests/posteriordb/reference_draws/` first
for `<posterior_name>.json.zip`. If found, it computes mean statistics from
those draws and uses them for validation. If no local zip is present, it falls
back to posteriordb's summary statistics.

## Contributing back to posteriordb (optional)

If you want to upstream reference draws:

1. Copy the generated files to a posteriordb checkout under
   `reference_posteriors/draws/`.
2. Open a PR against https://github.com/stan-dev/posteriordb.

## Notes

- The generator writes a temporary `.stan` file and compiled binary; both are
  ignored by `tests/posteriordb/reference_draws/.gitignore`.
- Sampling can take several minutes depending on model size.
