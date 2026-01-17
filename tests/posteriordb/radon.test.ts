import { describe, test, expect, beforeAll } from "vitest";
import { init, random } from "@jax-js/jax";
import { hmc, summary } from "jax-js-mcmc";
import {
  radonPooledModel,
  radonPooledData,
  radonHierarchicalModel,
  radonHierarchicalData,
} from "./models";

describe("radon pooled posteriordb", () => {
  beforeAll(async () => {
    await init("cpu");
  });

  test("logProb returns scalar", () => {
    const bound = radonPooledModel.bind(radonPooledData);
    const params = bound.initialParams();

    const lp = bound.logProb(params);
    expect(lp.shape).toEqual([]);
    const lpValue = lp.item();
    expect(typeof lpValue).toBe("number");
    expect(Number.isFinite(lpValue)).toBe(true);
  });

  test("posterior samples in reasonable range", async () => {
    const bound = radonPooledModel.bind(radonPooledData);

    const result = await hmc(bound.logProb, {
      numSamples: 100,
      numChains: 1,
      numWarmup: 50,
      initialParams: bound.initialParams(),
      key: random.key(42),
    });

    const stats = summary(result.draws);

    expect(stats.alpha.mean).toBeGreaterThan(-5);
    expect(stats.alpha.mean).toBeLessThan(10);
    expect(stats.beta.mean).toBeGreaterThan(-5);
    expect(stats.beta.mean).toBeLessThan(5);
    const sigmaConstrained = Math.exp(stats.sigma.mean);
    expect(sigmaConstrained).toBeGreaterThan(0);
    expect(sigmaConstrained).toBeLessThan(10);
  }, 120000);

  test("model compiles to complete state", () => {
    const bound = radonPooledModel.bind(radonPooledData);
    expect(bound.state).toBe("complete");
    expect(bound.paramNames).toContain("alpha");
    expect(bound.paramNames).toContain("beta");
    expect(bound.paramNames).toContain("sigma");
  });
});

// Note: Hierarchical model with indexing requires gather gradient support
// which is not yet implemented in jax-js. Skipping HMC tests for now.
describe("radon hierarchical posteriordb", () => {
  beforeAll(async () => {
    await init("cpu");
  });

  test("model compiles to complete state", () => {
    const bound = radonHierarchicalModel.bind(radonHierarchicalData);
    expect(bound.state).toBe("complete");
    expect(bound.paramNames).toContain("mu_alpha");
    expect(bound.paramNames).toContain("sigma_alpha");
    expect(bound.paramNames).toContain("alphaRaw");
    expect(bound.paramNames).toContain("beta");
    expect(bound.paramNames).toContain("sigma");
  });

  test.skip("posterior samples in reasonable range (requires gather gradient)", async () => {
    // Skipped: jax-js doesn't support gather gradient yet
  });
});
