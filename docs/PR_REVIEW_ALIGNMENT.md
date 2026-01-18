# PR Review Alignment: Columbia vs Yokohama

This document summarizes the agreements and differences between
`origin/StefanSko/columbia:docs/PR_REVIEW.md` and `docs/PR_REVIEW.md`.

## Agreements

- Recommend `origin/StefanSko/washington` as the merge candidate due to full
  posteriordb coverage.
- Treat missing posteriordb tests in `origin/codex/initial-implementation` as a
  blocker.
- Note that `origin/StefanSko/dakar` and `origin/StefanSko/melbourne` have
  incomplete posteriordb coverage and lack visualization support.

## Differences

- Type safety is overstated in Columbia for multiple branches:
  - `origin/StefanSko/washington` exposes `logProb` on predictive models and
    accepts partial observed data without error.
  - `origin/StefanSko/dakar` always includes `logProb` on `BoundModel`, so
    predictive models are not type-safe.
  - `origin/StefanSko/melbourne` returns a union from `bind` without compile-time
    enforcement.
- `origin/StefanSko/melbourne` posteriordb tests rely on hardcoded approximate
  summaries, not posteriordb reference data, which is a critical validation gap.
- `origin/StefanSko/dakar` does not implement named dimension tokens
  (`shape: "school"`), instead treating string shapes as data field names.
- `origin/codex/initial-implementation` does not allow parameter distributions to
  depend on model context, which blocks centered eight-schools and other
  hierarchical priors.
- Columbia flags missing JSDoc in washington as a merge blocker; Yokohama treats
  type-safety gaps and posteriordb gating as the blocking issues.

## Implications

- Washington is still the best merge base, but it must enforce
  complete/predictive typing and reject partial observed bindings.
- Posteriordb tests should be mandatory in CI for the recommended branch.
