# Open PR Review: jax-js-bayes Implementations

**Date:** 2026-01-18
**Reviewer:** Claude Opus 4.5
**Target:** docs/DESIGN.md specification with posteriordb validation

---

## Executive Summary

Four open PRs implement the jax-js-bayes library. All target the same DESIGN.md specification but differ significantly in completeness and validation coverage. **PR #5 (washington)** is recommended for merge due to complete posteriordb coverage. **PR #3 (codex)** has a critical gap: no posteriordb tests.

---

## PR Overview

| PR | Branch | Lines | posteriordb Models | Tests | Status |
|----|--------|-------|-------------------|-------|--------|
| #5 | washington | 1,843 | 11 (complete matrix) | ~50 | **Recommended** |
| #4 | dakar | 2,833 | 4 | 68 | Good |
| #2 | melbourne | 3,275 | 3 | 81 | Good |
| #3 | codex | 2,710 | 0 | ~35 | **Blocking defect** |

---

## Detailed Ratings

### 1. Architectural Design (1-10)

| PR | Score | Notes |
|----|-------|-------|
| #5 washington | 9 | Shape resolution system, context proxy for move semantics |
| #4 dakar | 9 | 5-stage compilation pipeline, excellent type-level enforcement |
| #2 melbourne | 9 | Runtime functional composition at bind() time |
| #3 codex | 7 | Solid foundation, less mature type system |

**Analysis:** PRs #2, #4, and #5 share a similar mature architecture:
- Declarative DSL with `param()`, `data()`, `observed()`, `model()`
- Complete/Predictive distinction via TypeScript conditional types
- Automatic Jacobian adjustment for constrained parameters
- Functional composition without hidden state

---

### 2. API Clarity (1-10)

| PR | Score | Notes |
|----|-------|-------|
| #4 dakar | 9 | Best PR documentation with usage examples |
| #2 melbourne | 9 | Clear primitives, named dimensions |
| #5 washington | 8 | Same API, missing JSDoc comments |
| #3 codex | 7 | Less documented |

**Common API pattern across all PRs:**
```typescript
const model = model({
  mu: param(normal(0, 5)),
  tau: param(halfCauchy(5), { constraint: positive() }),
  sigma: data({ shape: "school" }),
  y: observed(({ mu, sigma }) => normal(mu, sigma)),
});

const bound = model.bind({ y: [...], sigma: [...] });  // "complete"
const logP = bound.logProb({ mu, tau });
```

---

### 3. Type Safety (1-10)

| PR | Score | Notes |
|----|-------|-------|
| #4 dakar | 10 | `HasAllObserved<Spec, D>` conditional type |
| #5 washington | 9 | `BoundKind` + runtime discrimination |
| #2 melbourne | 9 | `BoundModel<S>` with conditional `logProb` |
| #3 codex | 8 | Discriminated unions, less sophisticated |

**Winner: #4 (dakar)** - Most advanced compile-time workflow enforcement prevents calling `logProb` on predictive models at the TypeScript level.

---

### 4. posteriordb Validation (Critical)

| PR | Score | Models Tested |
|----|-------|---------------|
| #5 washington | 10 | Eight Schools (C/NC), Kidscore, Kidscore Interaction, Radon (pooled/hierarchical), Wells, BLR, Log-Earn, Earn-Height, Mesquite |
| #4 dakar | 8 | Eight Schools, Kidscore, Radon (2), Wells |
| #2 melbourne | 8 | Eight Schools, Kidscore, Wells |
| #3 codex | 0 | **None** |

**Critical Note from CLAUDE.md:**
> "posteriordb tests validate inference correctness against Stan reference posteriors. These are the gold standard - if posteriordb tests fail, the library is producing wrong results."

**PR #3 fails this requirement entirely.**

---

### 5. Distribution & Constraint Coverage

| PR | Distributions | Constraints |
|----|---------------|-------------|
| #2 melbourne | 7 | 2 |
| #3 codex | 7 | 2 |
| #5 washington | 7 | 2 |
| #4 dakar | 6 (missing bernoulli) | 2 |

**Distributions implemented:**
- `normal`, `halfNormal`, `halfCauchy`, `exponential`, `uniform`
- `bernoulli`, `bernoulliLogit`

**Constraints implemented:**
- `positive()` - exp/log transform
- `bounded(low, high)` - sigmoid/logit transform

---

### 6. Code Quality (1-10)

| PR | Score | Notes |
|----|-------|-------|
| #2 melbourne | 9 | 81 tests, rigorous analytical validation |
| #5 washington | 9 | Best separation of concerns |
| #4 dakar | 9 | Consistent templates |
| #3 codex | 8 | Good patterns, less testing |

**Common patterns:**
- Pure functions, no mutation
- Proper jax-js `.ref` usage for move semantics
- Precomputed constants (LOG_2PI, etc.)
- TDD: logProb tests → sample tests → implementation

---

### 7. Memory Safety (1-10)

| PR | Score | Notes |
|----|-------|-------|
| #5 washington | 10 | `makeContextProxy()` abstraction |
| #4 dakar | 9 | Consistent `.ref` patterns |
| #3 codex | 9 | Explicit `owned[]` + `dispose()` |
| #2 melbourne | 9 | Sophisticated `.ref` usage |

All PRs correctly handle jax-js move semantics to prevent memory leaks in WebGPU/WASM environments.

---

### 8. Visualization Support

| PR | Score | Notes |
|----|-------|-------|
| #5 washington | 8 | tracePlot, densityPlot, pairPlot |
| #3 codex | 8 | Same, lazy-loads @observablehq/plot |
| #2 melbourne | 0 | No viz module |
| #4 dakar | 0 | No viz module |

---

## Overall Scores

| PR | Average | Recommendation |
|----|---------|----------------|
| **#5 washington** | **8.7** | **Merge** |
| #4 dakar | 7.8 | Good alternative |
| #2 melbourne | 8.1 | Good, needs posteriordb expansion |
| #3 codex | 6.3 | **Do not merge** (missing posteriordb) |

---

## Strengths by PR

### PR #5 (washington) - Recommended
- Complete 11-model posteriordb test matrix per DESIGN.md
- Best memory safety abstractions (`makeContextProxy`)
- Includes visualization module
- Shape resolution system with named dimensions

### PR #4 (dakar)
- Best TypeScript type-level modeling
- Excellent PR documentation with test plan
- Clear 5-stage compilation pipeline

### PR #2 (melbourne)
- Most tests (81 total)
- Rigorous analytical validation
- Clean runtime composition

### PR #3 (codex)
- Visualization support
- Solid DSL foundation
- **Critical gap: No inference validation**

---

## Gaps and Action Items

| PR | Required Before Merge |
|----|----------------------|
| #5 washington | Add JSDoc documentation to public functions |
| #4 dakar | Add viz module, add `bernoulli` distribution |
| #2 melbourne | Expand posteriordb to 11 models, add viz module |
| #3 codex | **Add posteriordb tests** (blocking) |

---

## Known Limitations (All PRs)

1. **Gather Gradient**: `np.take()` doesn't support gradients in jax-js. Radon hierarchical model compiles but HMC tests are skipped.

2. **Float32 Tolerances**: All tests use float32-appropriate tolerances (wider than float64) for WebGPU/WASM backends.

3. **No Performance Optimization**: No memoization or caching. Acceptable for v1.

4. **No Shape Broadcasting Validation**: Relies on user to align dimensions correctly.

---

## Conclusion

**PR #5 (washington)** should be merged. It is the only implementation with complete posteriordb coverage as specified in DESIGN.md, has solid architecture, proper memory safety, and includes visualization support.

**PR #3 (codex)** should not be merged in its current state due to missing posteriordb integration tests, which are explicitly required per project guidelines.

PRs #2 and #4 are viable alternatives but require additional work to match #5's posteriordb coverage.
