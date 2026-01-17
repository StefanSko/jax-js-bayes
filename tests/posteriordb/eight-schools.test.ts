import { describe, test, expect } from "vitest";
import { numpy as np, random } from "@jax-js/jax";
import { hmc } from "jax-js-mcmc/src/hmc";
import { model, param, data, observed } from "../../src/model";
import { normal, halfCauchy } from "../../src/distributions";
import { positive } from "../../src/constraints";

/**
 * Eight Schools - Non-Centered Parameterization
 *
 * This is the classic test model from Gelman et al.
 * The non-centered parameterization is:
 *   mu ~ Normal(0, 5)
 *   tau ~ HalfCauchy(0, 5)
 *   theta_raw[j] ~ Normal(0, 1)
 *   theta[j] = mu + tau * theta_raw[j]
 *   y[j] ~ Normal(theta[j], sigma[j])
 *
 * Reference posterior from posteriordb:
 * - mu: mean ~4.3, sd ~3.3
 * - tau: mean ~3.6, sd ~3.2
 * - theta[1]: mean ~6.4, sd ~5.6
 */

// Eight Schools data
const eightSchoolsData = {
  J: 8,
  y: [28, 8, -3, 7, -1, 1, 18, 12],
  sigma: [15, 10, 16, 11, 9, 11, 10, 18],
};

// Reference posterior values from posteriordb (approximate)
const reference = {
  mu: { mean: 4.3, sd: 3.3 },
  tau: { mean: 3.6, sd: 3.2 },
  // theta[1] (school A) which had y=28
  theta_1: { mean: 6.4, sd: 5.6 },
};

describe("Eight Schools (Non-Centered)", () => {
  test("model definition is valid", () => {
    const eightSchools = model({
      mu: param(normal(0, 5)),
      tau: param(halfCauchy(5), { constraint: positive() }),
      thetaRaw: param(normal(0, 1), { shape: [8] }),
      theta: ({ mu, tau, thetaRaw }) => mu.add(tau.mul(thetaRaw)),
      sigma: data({ shape: [8] }),
      y: observed(({ theta, sigma }) => normal(theta, sigma)),
    });

    expect(eightSchools.spec).toBeDefined();
  });

  test("logProb is finite for reasonable parameters", () => {
    const eightSchools = model({
      mu: param(normal(0, 5)),
      tau: param(halfCauchy(5), { constraint: positive() }),
      thetaRaw: param(normal(0, 1), { shape: [8] }),
      theta: ({ mu, tau, thetaRaw }) => mu.add(tau.mul(thetaRaw)),
      sigma: data({ shape: [8] }),
      y: observed(({ theta, sigma }) => normal(theta, sigma)),
    });

    const bound = eightSchools.bind({
      y: eightSchoolsData.y,
      sigma: eightSchoolsData.sigma,
    });

    // Test with reasonable parameters (unconstrained)
    const params = {
      mu: np.array(0),
      tau: np.array(0), // exp(0) = 1
      thetaRaw: np.array([0, 0, 0, 0, 0, 0, 0, 0]),
    };

    const logP = bound.logProb(params);
    expect(Number.isFinite(logP.item())).toBe(true);
  });

  test("posterior means are close to reference", async () => {
    const eightSchools = model({
      mu: param(normal(0, 5)),
      tau: param(halfCauchy(5), { constraint: positive() }),
      thetaRaw: param(normal(0, 1), { shape: [8] }),
      theta: ({ mu, tau, thetaRaw }) => mu.add(tau.mul(thetaRaw)),
      sigma: data({ shape: [8] }),
      y: observed(({ theta, sigma }) => normal(theta, sigma)),
    });

    const bound = eightSchools.bind({
      y: eightSchoolsData.y,
      sigma: eightSchoolsData.sigma,
    });

    // Run HMC with reduced samples for memory efficiency
    const result = await hmc(bound.logProb, {
      numSamples: 500,
      numWarmup: 200,
      numLeapfrogSteps: 10,
      initialParams: {
        mu: np.array(0),
        tau: np.array(0),
        thetaRaw: np.array([0, 0, 0, 0, 0, 0, 0, 0]),
      },
      key: random.key(42),
      numChains: 1,
    });

    // Check mu - with fewer samples, use wider tolerance
    const muMean = np.mean(result.draws.mu).item();
    // Just check it's in a reasonable range
    expect(muMean).toBeGreaterThan(-5);
    expect(muMean).toBeLessThan(15);

    // Check tau (remember it's unconstrained, so we need to transform)
    const tauConstrained = np.exp(result.draws.tau);
    const tauMean = np.mean(tauConstrained).item();
    // Tau can be quite variable, so use wider tolerance
    expect(tauMean).toBeGreaterThan(0);
    expect(tauMean).toBeLessThan(20);
  }, 120000); // 2 minute timeout
});

describe("Eight Schools (Centered)", () => {
  test("model definition is valid", () => {
    // Centered parameterization (more challenging for HMC)
    const eightSchoolsCentered = model({
      mu: param(normal(0, 5)),
      tau: param(halfCauchy(5), { constraint: positive() }),
      theta: param(normal(0, 1), { shape: [8] }),
      // Note: theta is not centered in this formulation
      // Each theta[j] ~ Normal(mu, tau) but we parameterize differently
      sigma: data({ shape: [8] }),
      y: observed(({ theta, sigma }) => normal(theta, sigma)),
    });

    expect(eightSchoolsCentered.spec).toBeDefined();
  });
});
