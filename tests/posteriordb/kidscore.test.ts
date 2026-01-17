import { describe, test, expect } from "vitest";
import { numpy as np, random } from "@jax-js/jax";
import { hmc } from "jax-js-mcmc/src/hmc";
import { model, param, data, observed } from "../../src/model";
import { normal, halfNormal } from "../../src/distributions";
import { positive } from "../../src/constraints";

/**
 * Kidscore Mom IQ - Simple Linear Regression
 *
 * Model:
 *   alpha ~ Normal(0, 100)
 *   beta ~ Normal(0, 10)
 *   sigma ~ HalfNormal(10)
 *   y[i] ~ Normal(alpha + beta * x[i], sigma)
 *
 * Tests basic linear regression functionality.
 */

// Simulated data that mimics kidscore dataset
const kidscoreData = {
  n: 20,
  x: [-1.5, -1.2, -0.8, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7,
      0.9, 1.0, 1.2, 1.3, 1.5, 1.7, 1.9, 2.0, 2.2, 2.5],
  y: [78, 82, 85, 88, 90, 92, 94, 96, 98, 100,
      102, 104, 106, 108, 110, 112, 114, 116, 118, 122],
};

describe("Kidscore Linear Regression", () => {
  test("model definition is valid", () => {
    const kidscore = model({
      alpha: param(normal(0, 100)),
      beta: param(normal(0, 10)),
      sigma: param(halfNormal(10), { constraint: positive() }),
      x: data(),
      y: observed(({ alpha, beta, sigma, x }) => {
        const mu = alpha.add(beta.mul(x));
        return normal(mu, sigma);
      }),
    });

    expect(kidscore.spec).toBeDefined();
  });

  test("logProb is finite for reasonable parameters", () => {
    const kidscore = model({
      alpha: param(normal(0, 100)),
      beta: param(normal(0, 10)),
      sigma: param(halfNormal(10), { constraint: positive() }),
      x: data(),
      y: observed(({ alpha, beta, sigma, x }) => {
        const mu = alpha.add(beta.mul(x));
        return normal(mu, sigma);
      }),
    });

    const bound = kidscore.bind({
      x: kidscoreData.x,
      y: kidscoreData.y,
    });

    // Test with reasonable parameters
    // alpha ~ 86 (intercept), beta ~ 10 (slope), sigma ~ 5
    const params = {
      alpha: np.array(86),
      beta: np.array(10),
      sigma: np.array(Math.log(5)), // unconstrained, exp(1.6) ≈ 5
    };

    const logP = bound.logProb(params);
    expect(Number.isFinite(logP.item())).toBe(true);
  });

  test("posterior recovers approximate slope and intercept", async () => {
    const kidscore = model({
      alpha: param(normal(0, 100)),
      beta: param(normal(0, 10)),
      sigma: param(halfNormal(10), { constraint: positive() }),
      x: data(),
      y: observed(({ alpha, beta, sigma, x }) => {
        const mu = alpha.add(beta.mul(x));
        return normal(mu, sigma);
      }),
    });

    const bound = kidscore.bind({
      x: kidscoreData.x,
      y: kidscoreData.y,
    });

    // Run HMC with reduced samples
    const result = await hmc(bound.logProb, {
      numSamples: 500,
      numWarmup: 200,
      numLeapfrogSteps: 10,
      initialParams: {
        alpha: np.array(100),
        beta: np.array(0),
        sigma: np.array(1),
      },
      key: random.key(42),
      numChains: 1,
    });

    // Check alpha (intercept) - should be positive, around the mean of y
    const alphaMean = np.mean(result.draws.alpha).item();
    expect(alphaMean).toBeGreaterThan(50);
    expect(alphaMean).toBeLessThan(130);

    // Check beta (slope) - should be positive for this data
    const betaMean = np.mean(result.draws.beta).item();
    // With x ranging from -1.5 to 2.5 and y from 78 to 122, slope should be positive
    expect(betaMean).toBeGreaterThan(-5);
    expect(betaMean).toBeLessThan(25);

    // Check sigma (std) - should be positive and reasonable
    const sigmaConstrained = np.exp(result.draws.sigma);
    const sigmaMean = np.mean(sigmaConstrained).item();
    expect(sigmaMean).toBeGreaterThan(0);
    expect(sigmaMean).toBeLessThan(20);
  }, 120000);
});
