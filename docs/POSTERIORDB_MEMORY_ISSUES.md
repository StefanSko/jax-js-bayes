# Posteriordb test memory issues

## Summary

`pnpm test tests/posteriordb` consistently fails with `ERR_WORKER_OUT_OF_MEMORY`
after the `eight-schools.smoke` test completes. This happens even when forcing
single-thread execution and increasing Node's heap limit to 4 GB.

## Environment

- Node: v25.2.1
- pnpm: 9.15.9
- OS: macOS 15.5 (arm64)
- Repo: `/tmp/washington` (branch `codex/merge-plan`)

## Reproduction

```bash
pnpm test tests/posteriordb
```

### Observed output (abridged)

```
RUN  v1.6.1 /private/tmp/washington

✓ tests/posteriordb/eight-schools.smoke.test.ts  (1 test) 59s

Unhandled Error
Error: Worker terminated due to reaching memory limit: JS heap out of memory
Serialized Error: { code: 'ERR_WORKER_OUT_OF_MEMORY' }
```

## Mitigations attempted

### Increase Node heap

```bash
NODE_OPTIONS=--max-old-space-size=4096 pnpm test tests/posteriordb
```

Result: still fails with `ERR_WORKER_OUT_OF_MEMORY`.

### Force single worker

```bash
NODE_OPTIONS=--max-old-space-size=4096 \
VITEST_MAX_THREADS=1 VITEST_MIN_THREADS=1 \
pnpm test tests/posteriordb
```

Result: still fails with `ERR_WORKER_OUT_OF_MEMORY`.

## Notes

- The error occurs after `eight-schools.smoke` passes; the second test file
  (`tests/posteriordb/posteriordb.test.ts`) appears to trigger the OOM.
- Local reference draw zips are loaded into memory to compute mean statistics.
  Some of these draw files are large (multi-chain JSON), which may contribute
  to heap pressure during test collection or execution.

## Next investigation steps

- Try a larger heap (e.g., `--max-old-space-size=8192`) if available.
- Run the posteriordb test file alone and measure memory usage.
- Consider switching vitest pools (threads vs forks) if supported by our config.
- Reduce peak memory by streaming draw parsing instead of fully loading JSON.
- Split posteriordb tests into smaller groups or run them serially with
  process-level isolation.
