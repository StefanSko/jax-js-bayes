# Root Cause Analysis: Memory Issues in jax-js-bayes

**Date:** 2026-01-18
**Analyzed by:** Claude (Agent)
**PRs investigated:** #5, #6

## Executive Summary

The memory issues in PR #6 are **real and expected** - they occur because posteriordb tests are now actually running, whereas in PR #5 they were silently skipped. The root cause is accumulation of jax-js Arrays during HMC sampling that aren't being freed fast enough by JavaScript's garbage collector.

## Key Finding: PR #5 Tests Were Skipped

**PR #5 did not actually run posteriordb tests.** The tests existed but were conditionally skipped:

```typescript
// PR #5 code
const eightStats = loadMeanStats("eight_schools-eight_schools_noncentered");
(eightStats ? test : test.skip)("eight schools noncentered", async () => {
  // test code
});
```

In PR #5:
- `loadMeanStats()` returned `null` if reference draws weren't found
- Tests were skipped when stats were unavailable
- No memory issues because tests didn't run

In PR #6:
- Added local reference draw files (large .json.zip files)
- `loadMeanStats()` now finds these files and returns data
- Changed to unconditional `test()` calls
- Tests actually run, revealing memory issues

## Root Cause: Array Accumulation in logProb

### The Problem

HMC sampling calls `logProb()` many times (e.g., 200 warmup + 400 samples × 2 chains = 1200 calls). Each `logProb` call:

1. **Creates a context** with JaxArray values
2. **Allocates temporary arrays** during computation
3. **Returns** a single scalar logProb value
4. **Leaves arrays** for JavaScript GC to clean up

### Memory Flow Analysis

#### src/compile.ts:compileLogProb (lines 44-106)

```typescript
return (params: Record<string, unknown>) => {
  let logProb = np.array(0);
  const ctx: ModelContext = {};
  const ctxProxy = makeContextProxy(ctx);

  // Transform params and store in ctx
  for (const name of paramNames) {
    const raw = toArray(params[name]);
    const rawRef = raw.ref;  // ref for constraint jacobian
    const constrained = defn.constraint
      ? defn.constraint.transform(raw)  // consumes raw
      : raw;
    ctx[name] = constrained;  // constrained has refCount=1
    // ...
  }

  // Compute derived values
  for (const derived of def.derived) {
    ctx[derived.name] = derived.fn(ctxProxy);  // stores new arrays in ctx
  }

  // Accumulate log probabilities
  for (const name of paramNames) {
    const dist = /* ... create or get distribution ... */;
    logProb = logProb.add(sumToScalar(dist.logProb(ctxProxy[name])));
    // ctxProxy[name] returns .ref, consumed by dist.logProb()
    // old logProb is consumed, new one created
  }

  return logProb;
  // ctx goes out of scope but arrays still have refCount=1
  // Arrays eligible for GC but not explicitly freed
};
```

**Issues identified:**

1. **ctx arrays persist**: Each array in `ctx` has `refCount=1` after logProb returns. They're eligible for GC but not explicitly freed.

2. **Proxy refs accumulate**: `makeContextProxy` returns `.ref` on each access, incrementing refCounts. These refs are consumed in computations, but the original `ctx` values remain.

3. **Derived arrays accumulate**: Derived values like `mu` in models are computed and stored in ctx with `refCount=1`.

4. **No explicit cleanup**: Unlike jax-js-mcmc which calls `treeDispose()` to explicitly free arrays, the compiled logProb function relies entirely on GC.

### Why This Matters

Over 1200 HMC iterations:
- Each iteration allocates ~10-50 arrays (params, derived values, intermediates)
- Each array might be 8-1000 elements × 4 bytes (float32)
- Total allocation: potentially 100+ MB per test
- JavaScript GC is non-deterministic and may not run frequently enough
- Workers have limited heap (even with 4GB, multiple tests exhaust it)

## Memory Management Patterns Comparison

### jax-js-mcmc (Correct Pattern)

```typescript
// hmc.ts:112
const logProbCurrent = logProb(treeClone(position));

// hmc.ts:137-141
if (accepted) {
  treeDispose(position);  // Explicit disposal
  position = proposalQ;
} else {
  treeDispose(proposalQ);  // Explicit disposal
}
```

**Key:** Explicit `treeClone()` and `treeDispose()` manage memory lifecycle.

### jax-js-bayes (Current Pattern)

```typescript
// No explicit disposal
// Relies on JavaScript GC
// Arrays accumulate until GC runs
```

**Issue:** No explicit memory management in the hot path (logProb).

## Why It Manifests as Worker OOM

From POSTERIORDB_MEMORY_ISSUES.md:

```
Error: Worker terminated due to reaching memory limit: JS heap out of memory
Serialized Error: { code: 'ERR_WORKER_OUT_OF_MEMORY' }
```

- Vitest runs tests in worker threads
- Each worker has a memory limit
- Multiple tests running sequentially in same worker
- Arrays from test 1 may not be freed before test 2 starts
- Worker heap fills up, OOM error

## Potential Solutions

### 1. Explicit Memory Management in logProb (Recommended)

Add explicit cleanup of ctx arrays after logProb computation:

```typescript
export function compileLogProb(/* ... */) {
  return (params: Record<string, unknown>) => {
    const ctx: ModelContext = {};
    try {
      // ... existing computation ...
      return logProb;
    } finally {
      // Explicitly free arrays in ctx
      for (const key in ctx) {
        const value = ctx[key];
        if (value && typeof value === 'object' && 'dispose' in value) {
          value.dispose();
        }
      }
    }
  };
}
```

**Pros:** Direct fix, follows jax-js-mcmc pattern
**Cons:** Requires understanding jax-js disposal API

### 2. Force GC Between Tests

```typescript
// vitest.config.ts
export default defineConfig({
  test: {
    poolOptions: {
      threads: {
        isolate: true,  // Force isolation
      },
    },
    maxConcurrency: 1,  // Run tests serially
  },
});
```

**Pros:** Simple config change
**Cons:** Slower tests, doesn't fix underlying issue

### 3. Split Tests Into Smaller Suites

Run posteriordb tests in separate worker processes:

```bash
pnpm test tests/posteriordb/eight-schools.test.ts
pnpm test tests/posteriordb/radon.test.ts
# etc
```

**Pros:** Isolates memory per test
**Cons:** Requires test reorganization, slower CI

### 4. Increase Worker Heap (Temporary)

```json
// package.json
{
  "scripts": {
    "test": "NODE_OPTIONS='--max-old-space-size=8192' vitest"
  }
}
```

**Pros:** Quick workaround
**Cons:** Doesn't scale, masks the problem

## Recommendations

1. **Short-term (PR #6):**
   - Document the memory issue clearly (already done via POSTERIORDB_MEMORY_ISSUES.md)
   - Consider splitting tests into separate files
   - Increase heap size temporarily

2. **Medium-term:**
   - Investigate explicit disposal in compileLogProb
   - Profile memory usage to confirm root cause
   - Consider if jax-js provides dispose/free APIs

3. **Long-term:**
   - Add memory management guidelines to CLAUDE.md
   - Create memory profiling tests
   - Consider whether jax-js needs better automatic memory management

## Additional Context

### jax-js Move Semantics

From JAX-JS-MEMORY.md:
- Every operation **consumes** its inputs
- Use `.ref` to keep arrays alive for multiple uses
- Arrays start with `refCount=1`
- When `refCount` reaches 0, array is freed

The current implementation correctly uses `.ref` through `makeContextProxy`, but doesn't explicitly dispose of arrays when done with them.

## Conclusion

**PR #5's "posteriordb coverage" was illusory** - tests were being skipped. PR #6 exposed the real memory issue by actually running the tests.

The memory problem is **real but solvable**. It's caused by:
1. HMC running 1000+ iterations per test
2. Each iteration allocating arrays that aren't explicitly freed
3. JavaScript GC not keeping up with allocation rate
4. Worker thread memory limits being exceeded

The fix requires either:
- Explicit memory management (preferred)
- Test isolation/splitting (workaround)
- Increased heap limits (temporary)

This is a **quality implementation blocker** - posteriordb tests are critical for validating inference correctness per DESIGN.md.
