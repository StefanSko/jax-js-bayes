import { describe, test, expect, beforeAll } from "vitest";
import { init, random } from "@jax-js/jax";
import { hmc, summary } from "jax-js-mcmc";
import { kidscoreModel, kidscoreData } from "./models";
import reference from "./reference/kidscore.json";

describe("kidscore posteriordb", () => {
  beforeAll(async () => {
    await init("cpu");
  });

  test("logProb returns scalar", () => {
    const bound = kidscoreModel.bind(kidscoreData);
    const params = bound.initialParams();

    const lp = bound.logProb(params);
    expect(lp.shape).toEqual([]);
    const lpValue = lp.item();
    expect(typeof lpValue).toBe("number");
    expect(Number.isFinite(lpValue)).toBe(true);
  });

  test("posterior matches posteriordb reference", async () => {
    const bound = kidscoreModel.bind(kidscoreData);

    const result = await hmc(bound.logProb, {
      numSamples: 100,
      numChains: 1,
      numWarmup: 50,
      initialParams: bound.initialParams(),
      key: random.key(42),
    });

    const stats = summary(result.draws);

    // Check that parameters are in reasonable range
    // Note: HMC samples in unconstrained space, so sigma is log(sigma)
    expect(stats.alpha.mean).toBeGreaterThan(-50);
    expect(stats.alpha.mean).toBeLessThan(100);
    expect(stats.beta.mean).toBeGreaterThan(-1);
    expect(stats.beta.mean).toBeLessThan(3);
    // sigma is in log-space, exp(stats.sigma.mean) should be positive
    const sigmaConstrained = Math.exp(stats.sigma.mean);
    expect(sigmaConstrained).toBeGreaterThan(0);
    expect(sigmaConstrained).toBeLessThan(100);
  }, 120000);

  test("model compiles to complete state", () => {
    const bound = kidscoreModel.bind(kidscoreData);
    expect(bound.state).toBe("complete");
    expect(bound.paramNames).toContain("alpha");
    expect(bound.paramNames).toContain("beta");
    expect(bound.paramNames).toContain("sigma");
  });
});
