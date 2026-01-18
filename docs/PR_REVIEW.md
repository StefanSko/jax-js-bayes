# PR Review: jax-js-bayes Implementations

Reviewed branches:
- origin/StefanSko/abu-dhabi
- origin/StefanSko/dakar
- origin/StefanSko/melbourne
- origin/StefanSko/washington
- origin/codex/initial-implementation

Reference: `docs/DESIGN.md`.

## Findings (ordered by severity)

### Critical
- No library implementation in `origin/StefanSko/abu-dhabi` (branch only adds docs/config; no `src/` or `tests/`).
- posteriordb coverage is missing or incomplete in three branches:
  - `origin/StefanSko/melbourne` uses approximate hardcoded summaries instead of posteriordb reference data (`tests/posteriordb/eight-schools.test.ts:19-37`).
  - `origin/StefanSko/dakar` covers only a subset of models (eight schools, kidscore, radon pooled/hierarchical, wells distance) and uses reduced datasets (`tests/posteriordb/models.ts:4-111`).
  - `origin/codex/initial-implementation` has no posteriordb test scripts (`package.json:13-17`).

### High
- Complete vs predictive type safety is not enforced in `origin/StefanSko/washington`:
  - `BoundModel` always exposes `logProb` regardless of state (`src/model.ts:40-46`).
  - `bind` accepts partial observed data without error, silently producing a predictive model with a callable `logProb` that ignores missing observations (`src/model.ts:213-238`).
- `origin/StefanSko/dakar` does not support named dimension tokens (e.g., `shape: "school"`). The implementation treats string shapes as data field names and only uses the first dimension (`src/compile.ts:136-163`).

### Medium
- `origin/StefanSko/dakar` returns unconstrained samples from `samplePrior`, while `model.simulate` expects unconstrained parameters; this conflicts with the design intent of `samplePrior` returning constrained draws used for simulation (`src/compile.ts:150-179`).
- `origin/StefanSko/dakar` is missing the `bernoulli` distribution required by the design (`src/distributions/index.ts:1-7`).
- `origin/StefanSko/melbourne` returns a union type from `bind` and allows partial observed data without error; type-level complete/predictive enforcement is not achieved (`src/model.ts:113-133`).
- `origin/codex/initial-implementation` does not support parameter distributions that depend on context, which blocks the centered eight-schools parameterization and other hierarchical priors (`src/model.ts:34-40`).
- `origin/StefanSko/dakar` uses `np.take` for hierarchical indexing with a known gradient limitation in jax-js, which can break HMC for models like radon hierarchical (`src/utils.ts:3-7`).

### Low
- `origin/StefanSko/dakar` and `origin/StefanSko/melbourne` use local file dependencies for `jax-js-mcmc` and `@jax-js/jax`, which is not suitable for publishable packages (`package.json:13-23` in each branch).
- Visualization helpers required by the design are missing in `origin/StefanSko/dakar` and `origin/StefanSko/melbourne` (no `src/viz/`).

## Posteriordb coverage

- `origin/StefanSko/washington`: All 11 target models with posteriordb data/means.
- `origin/StefanSko/dakar`: 5 models (eight schools centered/noncentered, kidscore, radon pooled/hierarchical, wells distance) using reduced datasets.
- `origin/StefanSko/melbourne`: 3 models (eight schools, kidscore, wells) with simulated data and approximate references.
- `origin/codex/initial-implementation`: None.
- `origin/StefanSko/abu-dhabi`: None.

## Comparison table (1-10)

| Branch | Architecture | API clarity | Type safety | Test coverage | Code quality | Performance |
| --- | --- | --- | --- | --- | --- | --- |
| origin/StefanSko/abu-dhabi | 1 | 1 | 1 | 1 | 2 | 1 |
| origin/StefanSko/dakar | 6 | 6 | 3 | 4 | 5 | 4 |
| origin/StefanSko/melbourne | 6 | 6 | 4 | 3 | 5 | 4 |
| origin/StefanSko/washington | 8 | 7 | 4 | 9 | 7 | 7 |
| origin/codex/initial-implementation | 7 | 7 | 4 | 3 | 7 | 7 |

## Gaps vs DESIGN.md

- `origin/StefanSko/abu-dhabi`: Entire implementation missing.
- `origin/StefanSko/dakar`: Missing `bernoulli`, missing viz, named dimension tokens unsupported, partial posteriordb coverage, `samplePrior`/`simulate` mismatch.
- `origin/StefanSko/melbourne`: No viz, missing posteriordb validation against reference data, incomplete type-level workflow enforcement, no named-dimension shape validation.
- `origin/StefanSko/washington`: Type-level complete/predictive enforcement missing; partial observed data not rejected.
- `origin/codex/initial-implementation`: No posteriordb tests; parameter priors cannot be context-dependent (blocks centered models); type-level workflow enforcement missing.

## Recommended PR

Recommend merging `origin/StefanSko/washington` after addressing the blocking issues below. It is the only branch with full posteriordb model coverage and closest alignment to the design.

## Blocking issues for merge

1) Enforce complete vs predictive at the type level (no `logProb` on predictive models) and reject partial observed data in `bind` (`src/model.ts:40-46`, `src/model.ts:213-238` in `origin/StefanSko/washington`).
2) Ensure posteriordb tests are mandatory in CI for the recommended branch (currently skipped when the local posteriordb dataset is missing; see `tests/posteriordb/pdb.ts` in `origin/StefanSko/washington`).
