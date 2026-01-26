# Agent Instructions for jax-js-bayes

## Project Overview

jax-js-bayes is a declarative Bayesian modeling library for jax-js.
It provides a TypeScript DSL for defining probabilistic models.

## Dependencies

This library depends on:
- **@jax-js/jax** - Array operations, autodiff
- **jax-js-mcmc-2** - HMC sampling (companion repo)

Clone references:

```bash
git clone https://github.com/ekzhang/jax-js.git /tmp/jax-js
git clone https://github.com/StefanSko/jax-js-mcmc-2.git /tmp/jax-js-mcmc-2
```

Refer to `/tmp/jax-js/src` for:
- Array API conventions
- How grad/jit/vmap work
- Random number generation patterns

See `docs/JAX-JS-MEMORY.md` for jax-js move semantics and `.ref` usage patterns.

## Types and Conventions

- `JsTree<Array>` means any nested object/array structure whose leaves are jax-js `Array` values.
- `logProb(params)` must return a scalar (0-dim) `Array` in float32 (not a JS number).
- Tests assume float32 behavior (WebGPU/WASM); tolerances reflect this.

## Development Workflow

### TDD for Distributions and Constraints

Each distribution/constraint is developed test-first:

1. Write logProb test against analytical formula
2. Write sample test checking mean/std
3. Implement distribution
4. Tests pass

For constraints, also test:
- transform/inverse roundtrip
- logDetJacobian against numerical differentiation

### Integration Testing: posteriordb

**CRITICAL:** posteriordb tests validate inference correctness against Stan reference posteriors. These are the gold standard - if posteriordb tests fail, the library is producing wrong results.

```bash
pnpm test tests/posteriordb
```

Must pass before merge. See `docs/DESIGN.md` for the 11 target models.

### Early Validation: posteriordb smoke test

Run a single light posteriordb model early in development as soon as the
DSL + compile path works.

- Recommended model: Eight Schools (noncentered)
- Suggested HMC config: `numSamples: 100`, `numWarmup: 100`, `numChains: 1`, `key: randomKey(0)`
- Goal: regression signal with loose tolerances, not full posterior precision

## JAX-JS Memory Model (Critical)

JAX-JS uses reference counting and move semantics. Every operation consumes
its inputs. If you need to reuse an array, take a `.ref` first.

```typescript
// WRONG - double consumption
const y = x.add(1);
const z = x.mul(2); // x already consumed

// CORRECT
const y = x.ref.add(1);
const z = x.mul(2); // x consumed here
```

### Testing memory correctness

```typescript
expect(oldState.position.refCount).toBe(0); // consumed
expect(newState.position.refCount).toBe(1); // fresh
expect(() => oldState.position.js()).toThrowError(ReferenceError);
```

See `docs/JAX-JS-MEMORY.md` for patterns and gotchas.

### Mandatory Integration Check (Memory)

When changing HMC wrappers, integrators, or inference loops, run the memory
profile from `jax-js-mcmc-2`:

```bash
cd /tmp/jax-js-mcmc-2
JAXJS_CACHE_LOG=1 NODE_OPTIONS="--expose-gc --loader ./tools/jaxjs-loader.mjs" \
  ITERATIONS=2000 LOG_EVERY=500 npx tsx examples/memory-profile-hmc-jit-step.ts
```

**Desired output:** memory should plateau (heap and rss do not trend upward) and
stay comfortably below ~300MB by the end of 2,000 iterations. If it grows
monotonically or exceeds ~500MB, treat as a regression.

You can also run it from this repo:

```bash
pnpm run memory:profile-mcmc2
```

**Why prefer JIT mode:** eager mode executes each primitive immediately and
allocates many intermediate arrays per step, pushing allocator high-water
marks upward. `jitStep()` fuses the whole step into a compiled kernel,
reducing intermediate allocations and stabilizing memory.

## Key Design Decisions

### Complete vs Predictive

The only type-level workflow enforcement. Keep it simple:

```typescript
type BoundModel<S extends "complete" | "predictive"> = { ... }
```

Don't add more workflow states - users manage their own workflow.

### Functional Composition

No special workflow functions. Just:
- `model.simulate(params)` - generate data
- `model.samplePrior()` - draw from prior
- `model.bind(data)` - bind data

Users compose these as needed.

### Observable Plot for Viz

Viz is optional. If user hasn't installed @observablehq/plot,
throw helpful error pointing to install command.

## Test Commands

Before running tests, install dependencies with `pnpm install`.

```bash
# Unit tests
pnpm test tests/distributions
pnpm test tests/constraints

# Integration tests
pnpm test tests/posteriordb

# All tests
pnpm test

# Browser tests
pnpm test:browser
```

## Code Style

- TypeScript strict mode
- Pure functions
- Match jax-js conventions
- Keep API surface small
