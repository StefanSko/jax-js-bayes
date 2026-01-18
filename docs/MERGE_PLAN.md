# Merge Plan: Best-of PRs into Washington

Base branch: `origin/StefanSko/washington`

Goal: merge the most complete implementation and enrich it with the best
features from other PRs while meeting `docs/DESIGN.md` and posteriordb
requirements.

## Phase 1: Establish the base and gaps

1) Create a worktree from `origin/StefanSko/washington`.
2) Inventory current gaps:
   - Type-level complete/predictive enforcement (no `logProb` on predictive).
   - Reject partial observed data in `bind`.
   - CI gating for posteriordb data (avoid skipping when missing).
3) Record candidate imports from other PRs:
   - melbourne: additional distribution/constraint tests and analytical checks.
   - dakar: type-level helpers and compile organization.
   - codex: memory ownership discipline (only where it fits jax-js `.ref`).

## Phase 2: Fix base blocking issues

1) Enforce `BoundModel<S>` so `logProb` is conditional on `S`.
2) Validate `bind` to reject partial observed data.
3) Make posteriordb tests mandatory in CI (fail if data missing).

## Phase 3: Selective feature imports

1) Port test improvements from melbourne:
   - Analytical logProb tests per distribution.
   - Sampling mean/std checks where appropriate.
2) Port compatible type helpers from dakar:
   - Conditional type utilities only if they align with washington structure.
3) Adopt codex memory patterns cautiously:
   - Keep `.ref` usage consistent with `docs/JAX-JS-MEMORY.md`.
   - Avoid explicit dispose if it conflicts with jax-js semantics.

## Phase 4: Verification and docs

1) Run unit tests for distributions and constraints.
2) Run full posteriordb suite (all 11 models).
3) Update docs with merge notes and any API deltas.

## Acceptance checklist

- All 11 posteriordb models pass without skips.
- Type system prevents `logProb` on predictive models.
- Partial observed data is rejected at bind time.
- Distributions/constraints tests cover logProb + sampling.
- Viz remains optional with a helpful install error message.
