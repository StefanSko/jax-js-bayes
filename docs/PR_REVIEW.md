# Open PR Review: jax-js-bayes Implementations

**Date:** 2026-01-18
**Reviewer:** Claude Opus 4.5
**Target:** docs/DESIGN.md specification with posteriordb validation
**Cross-referenced:** yokohama-v1 workspace review

---

## Executive Summary

Four open PRs implement the jax-js-bayes library. All target the same DESIGN.md specification but differ significantly in completeness and validation coverage. **PR #5 (washington)** is recommended for merge due to complete posteriordb coverage, but has **blocking type safety issues** that must be fixed first. **PR #3 (codex)** has a critical gap: no posteriordb tests.

---

## PR Overview

| PR | Branch | Lines | posteriordb Models | Tests | Status |
|----|--------|-------|-------------------|-------|--------|
| #5 | washington | 1,843 | 11 (complete matrix) | ~50 | **Recommended** |
| #4 | dakar | 2,833 | 4 | 68 | Good |
| #2 | melbourne | 3,275 | 3 | 81 | Good |
| #3 | codex | 2,710 | 0 | ~35 | **Blocking defect** |

---

## Critical Findings (by severity)

### Critical

| Issue | PR | Location | Description |
|-------|-----|----------|-------------|
| No posteriordb tests | #3 codex | `package.json:13-17` | Zero inference validation against Stan references |
| Type safety not enforced | #5 washington | `src/model.ts:40-46` | `BoundModel` always exposes `logProb` regardless of `kind` - should use conditional type |
| Partial observed accepted | #5 washington | `src/model.ts:213-238` | `bind()` silently accepts partial observed data, producing predictive model with callable `logProb` |
| Hardcoded references | #2 melbourne | `tests/posteriordb/eight-schools.test.ts:34-38` | Uses approximate hardcoded values instead of posteriordb reference data |

### High

| Issue | PR | Location | Description |
|-------|-----|----------|-------------|
| Named dimensions unsupported | #4 dakar | `src/compile.ts:136-163` | String shapes treated as data field names, only uses first dimension |
| Context-dependent priors blocked | #3 codex | `src/model.ts:34-40` | Cannot define priors that depend on other parameters (blocks centered parameterization) |
| Type-level enforcement missing | #2 melbourne | `src/model.ts:113-133` | Returns union type, no compile-time prevention of `logProb` on predictive |

### Medium

| Issue | PR | Location | Description |
|-------|-----|----------|-------------|
| samplePrior/simulate mismatch | #4 dakar | `src/compile.ts:150-179` | `samplePrior` returns unconstrained, design intent was constrained draws |
| Missing bernoulli | #4 dakar | `src/distributions/index.ts:1-7` | Required by DESIGN.md but not implemented |
| Gather gradient limitation | #4 dakar | `src/utils.ts:3-7` | `np.take` for hierarchical indexing breaks HMC gradients |

### Low

| Issue | PR | Location | Description |
|-------|-----|----------|-------------|
| Local file dependencies | #2, #4 | `package.json` | Uses `file:../` for jax-js-mcmc - not publishable |
| Missing viz module | #2, #4 | - | No `src/viz/` directory per DESIGN.md |

---

## Blocking Issues for Merge

**PR #5 (washington)** - recommended but requires fixes:
1. Enforce Complete/Predictive at type level: change `logProb` to `S extends "complete" ? ... : never` (`src/model.ts:40-46`)
2. Reject partial observed data in `bind()` or make `logProb` inaccessible (`src/model.ts:213-238`)
3. Ensure posteriordb tests are mandatory in CI (currently skip when data missing)

**PR #3 (codex)** - not mergeable:
1. Add posteriordb integration tests (critical requirement per CLAUDE.md)

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
| #4 dakar | 6 | Has conditional types but samplePrior/simulate workflow mismatch |
| #5 washington | 4 | `logProb` always exposed regardless of `kind` - **critical gap** |
| #2 melbourne | 4 | Returns union type, no compile-time `logProb` prevention |
| #3 codex | 4 | Discriminated unions but no workflow enforcement |

**Note:** All PRs fail to fully enforce Complete/Predictive at the type level. #4 (dakar) is closest with `HasAllObserved<Spec, D>` conditional type, but has other issues.

---

### 4. posteriordb Validation (Critical)

| PR | Score | Models Tested | Notes |
|----|-------|---------------|-------|
| #5 washington | 10 | 11 models (complete matrix) | Uses posteriordb reference data |
| #4 dakar | 6 | 5 models | Reduced datasets, subset coverage |
| #2 melbourne | 4 | 3 models | **Uses hardcoded approximate values, not posteriordb data** |
| #3 codex | 0 | **None** | No inference validation |

**Critical Note from CLAUDE.md:**
> "posteriordb tests validate inference correctness against Stan reference posteriors. These are the gold standard - if posteriordb tests fail, the library is producing wrong results."

**Issues:**
- PR #3 fails this requirement entirely
- PR #2 uses hardcoded reference values (`const reference = { mu: { mean: 4.3, sd: 3.3 } }`) instead of loading from posteriordb

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

## Overall Scores (Revised)

| PR | Architecture | API | Types | posteriordb | Quality | Memory | Viz | **Average** | Recommendation |
|----|--------------|-----|-------|-------------|---------|--------|-----|-------------|----------------|
| #5 washington | 8 | 8 | 4 | 10 | 9 | 10 | 8 | **8.1** | **Merge after fixes** |
| #4 dakar | 6 | 6 | 6 | 6 | 9 | 9 | 0 | **6.0** | Needs work |
| #2 melbourne | 6 | 6 | 4 | 4 | 9 | 9 | 0 | **5.4** | Needs work |
| #3 codex | 7 | 7 | 4 | 0 | 8 | 9 | 8 | **6.1** | **Do not merge** |

**Note:** Scores revised after cross-referencing with yokohama-v1 review which identified critical type safety gaps.

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
- Clean runtime composition
- **Weakness:** Uses hardcoded posteriordb values, not actual reference data

### PR #3 (codex)
- Visualization support
- Solid DSL foundation
- **Critical gap: No inference validation**

---

## Gaps and Action Items

| PR | Required Before Merge |
|----|----------------------|
| #5 washington | **Fix type safety** (`src/model.ts:40-46`), reject partial observed, add JSDoc |
| #4 dakar | Fix named dimensions, add viz module, add `bernoulli`, fix samplePrior |
| #2 melbourne | Replace hardcoded refs with posteriordb data, fix type safety, add viz |
| #3 codex | **Add posteriordb tests** (blocking), add context-dependent priors |

---

## Known Limitations (All PRs)

1. **Gather Gradient**: `np.take()` doesn't support gradients in jax-js. Radon hierarchical model compiles but HMC tests are skipped.

2. **Float32 Tolerances**: All tests use float32-appropriate tolerances (wider than float64) for WebGPU/WASM backends.

3. **No Performance Optimization**: No memoization or caching. Acceptable for v1.

4. **No Shape Broadcasting Validation**: Relies on user to align dimensions correctly.

---

## Conclusion

**PR #5 (washington)** is recommended for merge after addressing blocking issues:
1. Fix type safety: `logProb` must be conditional on `kind === "complete"` (`src/model.ts:40-46`)
2. Reject partial observed data or hide `logProb` on predictive models (`src/model.ts:213-238`)
3. Make posteriordb tests mandatory in CI

It is the only implementation with complete posteriordb coverage as specified in DESIGN.md, has solid architecture, proper memory safety, and includes visualization support.

**PR #3 (codex)** should not be merged due to missing posteriordb integration tests.

**PRs #2 and #4** require significant work: melbourne uses hardcoded reference values, dakar has workflow mismatches and missing features.

---

## Appendix: Cross-Reference with yokohama-v1 Review

This review was updated after comparing with the yokohama-v1 workspace review, which identified several critical issues missed in the initial assessment:
- Washington type safety gap (originally rated 9/10, revised to 4/10)
- Melbourne hardcoded posteriordb references (originally rated 8/10, revised to 4/10)
- Dakar samplePrior/simulate workflow mismatch
- Codex context-dependent prior limitation
