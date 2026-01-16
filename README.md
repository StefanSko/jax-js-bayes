# jax-js-bayes

Declarative Bayesian modeling library for [jax-js](https://github.com/ekzhang/jax-js).

## Status

**Work in progress** - See [docs/DESIGN.md](docs/DESIGN.md) for the full design document.

## Overview

A TypeScript-native DSL for defining probabilistic models, with inference powered by [jax-js-mcmc](https://github.com/StefanSko/jax-js-mcmc).

```typescript
import { model, param, observed, data } from "jax-js-bayes";
import { normal, halfCauchy } from "jax-js-bayes/distributions";
import { positive } from "jax-js-bayes/constraints";
import { hmc } from "jax-js-mcmc";

const eightSchools = model({
  mu: param(normal(0, 5)),
  tau: param(halfCauchy(5), { constraint: positive() }),
  thetaRaw: param(normal(0, 1), { shape: "school" }),
  theta: ({ mu, tau, thetaRaw }) => mu.add(tau.mul(thetaRaw)),
  sigma: data({ shape: "school" }),
  y: observed(({ theta, sigma }) => normal(theta, sigma)),
});

const bound = eightSchools.bind({ y: [...], sigma: [...] });
const result = await hmc(bound.logProb, { numSamples: 4000 });
```

## Features (Planned)

- TypeScript-native modeling DSL
- Distributions: Normal, HalfNormal, HalfCauchy, Exponential, Bernoulli
- Constraints: Positive, Bounded (with automatic Jacobian adjustment)
- Visualizations via Observable Plot
- Validated against 11 posteriordb reference posteriors

## Related

- [jax-js](https://github.com/ekzhang/jax-js) - JAX in the browser
- [jax-js-mcmc](https://github.com/StefanSko/jax-js-mcmc) - HMC sampling library

## License

MIT
