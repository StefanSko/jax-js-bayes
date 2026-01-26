# Posteriordb test memory issues

## Summary

`pnpm test tests/posteriordb` still fails with `ERR_WORKER_OUT_OF_MEMORY`.
We fixed the HMC eager-memory leak via a JIT warmup path, but the posteriordb
suite itself still OOMs due to high peak memory in the test harness.

## Environment (Jan 26, 2026)

- Node: v25.2.1
- pnpm: 9.15.9
- OS: macOS 15.5 (arm64)
- Repo: `/Users/stefansko/conductor/workspaces/jax-js-bayes/belo-horizonte-v1`

## Reproduction

```bash
pnpm test tests/posteriordb
```

### Observed output (abridged)

```
RUN  v1.6.1 /Users/stefansko/conductor/workspaces/jax-js-bayes/belo-horizonte-v1

Unhandled Error
Error: Worker exited unexpectedly
FATAL ERROR: Ineffective mark-compacts near heap limit Allocation failed - JavaScript heap out of memory
```

## What we changed / instrumented

### 1) HMC eager-memory leak isolated and fixed (local wiring)

- Added a memory repro test (`tests/memory/hmc-repro.test.ts`) showing
  linear growth in eager mode even with trivial log density.
- Confirmed memory stability with JIT (`HMC_FORCE_JIT=1`) even with warmup.
- Patched `/tmp/jax-js-mcmc-2` to support dynamic step size as an Array and
  `stepWithSize(...)` so warmup can stay JIT-compiled.
- Wired `jax-js-bayes` to use `stepWithSize` when available.

Result: memory is flat for the repro when JIT is forced on, even with warmup.

### 2) Posteriordb memory instrumentation

Added optional memory logging and disposal hooks (guarded by env vars):

- `tests/posteriordb/memory.ts`
- `tests/posteriordb/pdb.ts` logging around zip parsing
- `tests/posteriordb/posteriordb.test.ts` and `eight-schools.smoke.test.ts`
  dispose of draws and log heap/rss when enabled

### 3) Process isolation attempt

Added `tools/run-posteriordb-isolated.sh` and script:

```bash
pnpm run test:posteriordb:isolated
```

This runs `eight-schools.smoke` and `posteriordb.test.ts` in separate
processes with an 8GB heap. `posteriordb.test.ts` still OOMs (~7.2GB heap).

## Current state

- HMC warmup memory leak is resolved when using JIT warmup with dynamic step size.
- Posteriordb still OOMs due to high peak memory in the test harness.

## Likely cause (posteriordb)

- `loadLocalMeanStats()` loads full reference draw zips into memory and
  computes means by fully materializing arrays.
- The heavy posteriordb test file appears to trigger OOM even in isolation.

## Next steps (planned)

1) **Stream reference draw parsing** to avoid loading full JSON arrays.
   - Compute means incrementally (single pass) instead of materializing all draws.
2) **Reduce reference draw size/usage**:
   - Only compute means for parameters needed by the current test.
3) **Keep process isolation** as a fallback for CI.
4) **Upstream `jax-js-mcmc-2` patch** (dynamic step size + JIT warmup)
   and remove the local tsconfig override once released.
