import { describe, test, expect } from "vitest";
import { numpy as np, random } from "@jax-js/jax";
import { hmc } from "jax-js-mcmc/src/hmc";
import { model, param, data, observed } from "../../src/model";
import { normal } from "../../src/distributions";
import { bernoulliLogit } from "../../src/distributions/bernoulliLogit";

/**
 * Wells Distance - Logistic Regression
 *
 * Model:
 *   alpha ~ Normal(0, 5)
 *   beta ~ Normal(0, 2.5)
 *   y[i] ~ Bernoulli(sigmoid(alpha + beta * x[i]))
 *
 * Tests logistic regression with bernoulliLogit.
 */

// Simulated data for logistic regression
const wellsData = {
  n: 30,
  // Normalized distance values
  x: [-2.0, -1.8, -1.5, -1.3, -1.0, -0.8, -0.5, -0.3, -0.1, 0.0,
      0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8,
      2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8],
  // Binary outcome (switched wells or not) - higher x -> more likely to switch
  y: [0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
      1, 1, 0, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
};

describe("Wells Logistic Regression", () => {
  test("model definition is valid", () => {
    const wells = model({
      alpha: param(normal(0, 5)),
      beta: param(normal(0, 2.5)),
      x: data(),
      y: observed(({ alpha, beta, x }) => {
        const logit = alpha.add(beta.mul(x));
        return bernoulliLogit(logit);
      }),
    });

    expect(wells.spec).toBeDefined();
  });

  test("logProb is finite for reasonable parameters", () => {
    const wells = model({
      alpha: param(normal(0, 5)),
      beta: param(normal(0, 2.5)),
      x: data(),
      y: observed(({ alpha, beta, x }) => {
        const logit = alpha.add(beta.mul(x));
        return bernoulliLogit(logit);
      }),
    });

    const bound = wells.bind({
      x: wellsData.x,
      y: wellsData.y,
    });

    // Test with reasonable parameters
    const params = {
      alpha: np.array(0),
      beta: np.array(1),
    };

    const logP = bound.logProb(params);
    expect(Number.isFinite(logP.item())).toBe(true);
  });

  test("posterior recovers positive slope", async () => {
    const wells = model({
      alpha: param(normal(0, 5)),
      beta: param(normal(0, 2.5)),
      x: data(),
      y: observed(({ alpha, beta, x }) => {
        const logit = alpha.add(beta.mul(x));
        return bernoulliLogit(logit);
      }),
    });

    const bound = wells.bind({
      x: wellsData.x,
      y: wellsData.y,
    });

    // Run HMC with reduced samples
    const result = await hmc(bound.logProb, {
      numSamples: 500,
      numWarmup: 200,
      numLeapfrogSteps: 10,
      initialParams: {
        alpha: np.array(0),
        beta: np.array(0),
      },
      key: random.key(42),
      numChains: 1,
    });

    // Check beta (slope) - should be positive since higher x -> y=1
    const betaMean = np.mean(result.draws.beta).item();
    expect(betaMean).toBeGreaterThan(0);
  }, 120000);
});
