# jax-js-bayes Design Document

**Date:** 2026-01-16
**Status:** Draft
**Repository:** github.com/StefanSko/jax-js-bayes
**Depends on:** [jax-js-mcmc](./2026-01-16-jax-js-mcmc-design.md)

## Overview

A declarative Bayesian modeling library for jax-js. Provides a TypeScript-native DSL for defining probabilistic models, compiles them to log probability functions, and uses jax-js-mcmc for inference.

This is a standalone JavaScript library - not a port of any Python library. Designed for JS developers who want to do Bayesian inference in the browser.

## Goals

1. **JS-native** - TypeScript API that feels natural to JS developers
2. **Simple** - Low friction, minimal boilerplate
3. **Composable** - Small functions that combine naturally
4. **Type-safe** - Complete/Predictive distinction at compile time
5. **Validated** - Tested against posteriordb reference posteriors

## Non-Goals (v1)

- Full Bayesian workflow enforcement (just Complete/Predictive)
- Python export / IR format (reserved for v2)
- All distributions (just enough for 11 posteriordb models)

## Relationship to jax-js-mcmc

```
@jax-js/jax (external)
     ↑
jax-js-mcmc (HMC sampler - see companion design doc)
     ↑
jax-js-bayes (this library)
```

jax-js-bayes compiles models to `logProb` functions that jax-js-mcmc can sample:

```typescript
// jax-js-bayes creates the model
const model = defineModel({ ... });
const bound = model.bind(data);

// jax-js-bayes compiles to logProb
const logProb = bound.logProb;  // (params) => scalar

// jax-js-mcmc samples it
const result = await hmc(logProb, { ... });
```

## API Design

### Model Definition

```typescript
import { model, param, observed, data } from "jax-js-bayes";
import { normal, halfCauchy } from "jax-js-bayes/distributions";
import { positive } from "jax-js-bayes/constraints";

const eightSchools = model({
  // Parameters with priors
  mu: param(normal(0, 5)),
  tau: param(halfCauchy(5), { constraint: positive() }),
  thetaRaw: param(normal(0, 1), { shape: "school" }),

  // Derived quantities (arrow functions)
  theta: ({ mu, tau, thetaRaw }) => mu.add(tau.mul(thetaRaw)),

  // Data declarations
  sigma: data({ shape: "school" }),

  // Likelihood
  y: observed(({ theta, sigma }) => normal(theta, sigma)),
});
```

### Data Binding

```typescript
// Complete: all data including observations
const complete = eightSchools.bind({
  y: [28, 8, -3, 7, -1, 1, 18, 12],
  sigma: [15, 10, 16, 11, 9, 11, 10, 18],
});
// Type: BoundModel<"complete">

// Predictive: covariates only, no observations
const predictive = eightSchools.bind({
  sigma: [15, 10, 16, 11, 9, 11, 10, 18],
});
// Type: BoundModel<"predictive">
```

### Type-Safe Inference

```typescript
import { hmc } from "jax-js-mcmc";

// sample() only accepts complete models
const result = await hmc(complete.logProb, { numSamples: 4000 });  // OK
const result = await hmc(predictive.logProb, { ... });  // Type error

// samplePrior works with both
const priorDraws = await samplePrior(predictive);  // OK
const priorDraws = await samplePrior(complete);    // OK
```

### Simulation (Generative Model)

```typescript
// Draw parameters from prior
const priorParams = model.samplePrior({ key: randomKey(42) });
// { mu: 2.3, tau: 1.1, thetaRaw: [0.2, -0.5, ...] }

// Simulate data from parameters
const fakeData = model.simulate(priorParams, { key: randomKey(43) });
// { y: [25, 12, ...], sigma: [15, 10, ...] }

// Normal workflow on simulated data
const bound = model.bind(fakeData);
const result = await hmc(bound.logProb, { numSamples: 1000 });

// Check recovery (user does this themselves)
console.log(result.summary());  // Compare to priorParams
```

### Results Access

```typescript
const result = await hmc(complete.logProb, { numSamples: 4000, numChains: 4 });

// Raw draws
result.draws.mu;        // Array [chains, samples]
result.draws.tau;       // Array [chains, samples]
result.draws.theta;     // Array [chains, samples, schools]

// Summary statistics (from jax-js-mcmc)
import { summary } from "jax-js-mcmc/diagnostics";
summary(result.draws);
// { mu: { mean, sd, q5, q95, rhat, ess }, ... }
```

## Distributions

### v1 Scope

Enough for 11 posteriordb models:

```typescript
// Continuous
import {
  normal,       // Normal(loc, scale)
  halfNormal,   // |Normal(0, scale)| - positive support
  halfCauchy,   // |Cauchy(0, scale)| - positive support, heavier tails
  exponential,  // Exponential(rate) - positive support
  uniform,      // Uniform(low, high) - bounded support
} from "jax-js-bayes/distributions";

// Discrete
import {
  bernoulli,       // Bernoulli(p) - p in [0,1]
  bernoulliLogit,  // Bernoulli(sigmoid(logit)) - logit in (-∞, ∞)
} from "jax-js-bayes/distributions";
```

### Distribution Interface

```typescript
interface Distribution {
  logProb(x: Array): Array;
  sample(key: PRNGKey, shape?: number[]): Array;
}

// Example implementation
function normal(loc: number | Array, scale: number | Array): Distribution {
  return {
    logProb(x) {
      const z = x.sub(loc).div(scale);
      return z.pow(2).mul(-0.5).sub(np.log(scale)).sub(0.5 * Math.log(2 * Math.PI));
    },
    sample(key, shape = []) {
      return random.normal(key, shape).mul(scale).add(loc);
    },
  };
}
```

### HalfNormal / HalfCauchy

Implemented via absolute value:

```typescript
function halfNormal(scale: number | Array): Distribution {
  const base = normal(0, scale);
  return {
    logProb(x) {
      // log(2) + Normal(0, scale).logProb(x) for x > 0
      return np.log(2).add(base.logProb(x));
    },
    sample(key, shape = []) {
      return np.abs(base.sample(key, shape));
    },
  };
}
```

## Constraints

### v1 Scope

```typescript
import { positive, bounded } from "jax-js-bayes/constraints";

// Positive: exp transform
param(halfCauchy(5), { constraint: positive() })

// Bounded: sigmoid transform
param(uniform(0, 1), { constraint: bounded(0, 1) })
```

### Constraint Interface

```typescript
interface Constraint {
  transform(unconstrained: Array): Array;    // unconstrained → constrained
  inverse(constrained: Array): Array;        // constrained → unconstrained
  logDetJacobian(unconstrained: Array): Array;  // log|det(J)|
}

// Example: positive constraint via exp
function positive(): Constraint {
  return {
    transform: (x) => np.exp(x),
    inverse: (y) => np.log(y),
    logDetJacobian: (x) => x,  // d/dx exp(x) = exp(x), log|exp(x)| = x
  };
}

// Example: bounded constraint via sigmoid
function bounded(low: number, high: number): Constraint {
  const range = high - low;
  return {
    transform: (x) => np.sigmoid(x).mul(range).add(low),
    inverse: (y) => logit(y.sub(low).div(range)),
    logDetJacobian: (x) => {
      const s = np.sigmoid(x);
      return np.log(s).add(np.log(1 - s)).add(Math.log(range));
    },
  };
}
```

### Automatic Jacobian Adjustment

Model compilation automatically adds Jacobian corrections:

```typescript
// User writes:
param(halfCauchy(5), { constraint: positive() })

// Compiled logProb includes:
// prior.logProb(transform(unconstrained)) + logDetJacobian(unconstrained)
```

## Visualizations

### Built on Observable Plot

```typescript
import { tracePlot, densityPlot, pairPlot } from "jax-js-bayes/viz";

// Trace plot
const el = tracePlot(result.draws.mu);
document.getElementById("plot").appendChild(el);

// Density with prior overlay
densityPlot(result.draws.mu, { prior: normal(0, 5) });

// Pair plot for correlations
pairPlot(result.draws, { params: ["mu", "tau"] });
```

### Optional Dependency

Observable Plot is an optional peer dependency:

```json
{
  "peerDependencies": {
    "@observablehq/plot": ">=0.6"
  },
  "peerDependenciesMeta": {
    "@observablehq/plot": { "optional": true }
  }
}
```

If not installed, viz functions throw helpful error:

```typescript
// If @observablehq/plot not installed:
tracePlot(draws);
// Error: "jax-js-bayes/viz requires @observablehq/plot. Install with: npm i @observablehq/plot"
```

## Project Structure

```
jax-js-bayes/
├── src/
│   ├── model.ts              # model(), param(), observed(), data()
│   ├── compile.ts            # Compile model to logProb function
│   ├── distributions/
│   │   ├── normal.ts
│   │   ├── halfNormal.ts
│   │   ├── halfCauchy.ts
│   │   ├── exponential.ts
│   │   ├── uniform.ts
│   │   ├── bernoulli.ts
│   │   └── index.ts
│   ├── constraints/
│   │   ├── positive.ts
│   │   ├── bounded.ts
│   │   └── index.ts
│   ├── viz/
│   │   ├── trace.ts
│   │   ├── density.ts
│   │   ├── pairs.ts
│   │   └── index.ts
│   └── index.ts
├── tests/
│   ├── model.test.ts
│   ├── distributions/
│   │   └── *.test.ts
│   ├── constraints/
│   │   └── *.test.ts
│   └── posteriordb/          # Integration tests
│       ├── models.ts
│       ├── eight-schools.test.ts
│       ├── kidscore.test.ts
│       └── ...
├── CLAUDE.md
├── package.json
└── tsconfig.json
```

## Testing Strategy

### Unit Tests

Each distribution and constraint tested in isolation:

```typescript
// tests/distributions/normal.test.ts
describe("normal distribution", () => {
  test("logProb matches analytical formula", () => {
    const d = normal(0, 1);
    expect(d.logProb(np.array([0]))).toBeCloseTo(-0.9189, 4);  // -0.5*log(2π)
  });

  test("samples have correct mean and std", async () => {
    const d = normal(2, 3);
    const samples = d.sample(randomKey(42), [10000]);
    expect(mean(samples)).toBeCloseTo(2, { tolerance: 0.1 });
    expect(std(samples)).toBeCloseTo(3, { tolerance: 0.1 });
  });
});
```

### Constraint Tests

```typescript
// tests/constraints/positive.test.ts
describe("positive constraint", () => {
  const c = positive();

  test("transform maps R to R+", () => {
    expect(c.transform(np.array([0]))).toBeCloseTo(1);
    expect(c.transform(np.array([-1]))).toBeCloseTo(0.368);
    expect(c.transform(np.array([1]))).toBeCloseTo(2.718);
  });

  test("inverse is correct", () => {
    const x = np.array([0.5, 1.0, 2.0]);
    const y = c.transform(x);
    expect(c.inverse(y)).toBeCloseTo(x);
  });

  test("logDetJacobian is correct", () => {
    // Numerical check: compare to finite difference
    const x = np.array([0.5]);
    const analytical = c.logDetJacobian(x);
    const numerical = numericalLogDetJacobian(c.transform, x);
    expect(analytical).toBeCloseTo(numerical, { tolerance: 1e-5 });
  });
});
```

### Integration Tests: posteriordb

Test full models against reference posteriors:

```typescript
// tests/posteriordb/eight-schools.test.ts
import { eightSchoolsModel, eightSchoolsData } from "./models";
import reference from "./reference/eight-schools.json";

describe("eight schools", () => {
  test("posterior matches posteriordb reference", async () => {
    const bound = eightSchoolsModel.bind(eightSchoolsData);
    const result = await hmc(bound.logProb, {
      numSamples: 8000,
      numChains: 4,
      key: randomKey(42),
    });

    const stats = summary(result.draws);

    expect(stats.mu.mean).toBeCloseTo(reference.mu.mean, { tolerance: 0.15 });
    expect(stats.tau.mean).toBeCloseTo(reference.tau.mean, { tolerance: 0.15 });
  });
});
```

### Test Matrix: 11 posteriordb Models

| Model | Distributions | Constraints | Validates |
|-------|---------------|-------------|-----------|
| Eight Schools (NC) | Normal, HalfCauchy | Positive | Hierarchical, non-centered |
| Eight Schools (C) | Normal, HalfCauchy | Positive | Centered parameterization |
| Kidscore Mom IQ | Normal, HalfNormal | Positive | Linear regression |
| Radon Pooled | Normal, HalfNormal | Positive | Pooled regression |
| Radon Hierarchical | Normal, HalfNormal | Positive | Multi-level indexing |
| Wells Distance | BernoulliLogit, Normal | None | Logistic regression |
| BLR | Normal, HalfNormal | Positive | Matrix operations |
| Log Earnings | Normal, HalfNormal | Positive | Log transforms |
| Earnings Height | Normal, HalfNormal | Positive | Wide priors |
| Kidscore Interaction | Normal, HalfCauchy | Positive | Interaction terms |
| Mesquite Log Volume | Normal, HalfNormal | Positive | Multiple predictors |

## Dependencies

```json
{
  "name": "jax-js-bayes",
  "version": "0.1.0",
  "dependencies": {
    "jax-js-mcmc": "^0.1.0"
  },
  "peerDependencies": {
    "@jax-js/jax": ">=0.1.0"
  },
  "peerDependenciesMeta": {
    "@observablehq/plot": { "optional": true }
  },
  "devDependencies": {
    "vitest": "^1.0.0",
    "playwright": "^1.40.0",
    "@observablehq/plot": "^0.6.0"
  }
}
```

## CLAUDE.md (Agent Instructions)

```markdown
# Agent Instructions for jax-js-bayes

## Project Overview

jax-js-bayes is a declarative Bayesian modeling library for jax-js.
It provides a TypeScript DSL for defining probabilistic models.

## Dependencies

This library depends on:
- **@jax-js/jax** - Array operations, autodiff
- **jax-js-mcmc** - HMC sampling (see companion repo)

Clone references:

\`\`\`bash
git clone https://github.com/ekzhang/jax-js.git /tmp/jax-js
git clone https://github.com/StefanSko/jax-js-mcmc.git /tmp/jax-js-mcmc
\`\`\`

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

### Integration Testing

After unit tests pass, run posteriordb integration tests:

\`\`\`bash
pnpm test tests/posteriordb
\`\`\`

These compare against reference posteriors. Must pass before merge.

## Key Design Decisions

### Complete vs Predictive

The only type-level workflow enforcement. Keep it simple:

\`\`\`typescript
type BoundModel<S extends "complete" | "predictive"> = { ... }
\`\`\`

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

\`\`\`bash
# Unit tests
pnpm test tests/distributions
pnpm test tests/constraints

# Integration tests
pnpm test tests/posteriordb

# All tests
pnpm test

# Browser tests
pnpm test:browser
\`\`\`

## Code Style

- TypeScript strict mode
- Pure functions
- Match jax-js conventions
- Keep API surface small
```

## Implementation Order

1. **Model DSL** - `model()`, `param()`, `observed()`, `data()`
2. **Distributions** - Normal, HalfNormal, HalfCauchy, Exponential, Uniform, Bernoulli
3. **Constraints** - Positive, Bounded
4. **Compilation** - Model → logProb function
5. **Binding** - Complete/Predictive type distinction
6. **Simulation** - `samplePrior()`, `simulate()`
7. **posteriordb tests** - Validate against references
8. **Visualizations** - Observable Plot wrappers

## Future (v2+)

- More distributions (Gamma, Poisson, Binomial, StudentT)
- Shared IR format with jaxstan for Python export
- Formula syntax (`y ~ x + (1|group)`)
- Interactive playground UI
