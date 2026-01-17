# Agent Instructions for jax-js-bayes

## Project Overview

jax-js-bayes is a declarative Bayesian modeling library for jax-js.
It provides a TypeScript DSL for defining probabilistic models.

## Dependencies

This library depends on:
- **@jax-js/jax** - Array operations, autodiff
- **jax-js-mcmc** - HMC sampling (see companion repo)

Clone references:

```bash
git clone https://github.com/ekzhang/jax-js.git /tmp/jax-js
git clone https://github.com/StefanSko/jax-js-mcmc.git /tmp/jax-js-mcmc
```

Refer to /tmp/jax-js/src for:
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
