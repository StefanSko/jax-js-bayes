import { describe, test, expect, beforeAll } from "vitest";
import { init, random } from "@jax-js/jax";
import { hmc, summary } from "jax-js-mcmc";
import { wellsDistanceModel, wellsDistanceData } from "./models";

describe("wells distance posteriordb", () => {
  beforeAll(async () => {
    await init("cpu");
  });

  test("logProb returns scalar", () => {
    const bound = wellsDistanceModel.bind(wellsDistanceData);
    const params = bound.initialParams();

    const lp = bound.logProb(params);
    expect(lp.shape).toEqual([]);
    const lpValue = lp.item();
    expect(typeof lpValue).toBe("number");
    expect(Number.isFinite(lpValue)).toBe(true);
  });

  test("posterior samples in reasonable range", async () => {
    const bound = wellsDistanceModel.bind(wellsDistanceData);

    const result = await hmc(bound.logProb, {
      numSamples: 100,
      numChains: 1,
      numWarmup: 50,
      initialParams: bound.initialParams(),
      key: random.key(42),
    });

    const stats = summary(result.draws);

    // Logistic regression coefficients - wider tolerance for float32 variability
    expect(stats.alpha.mean).toBeGreaterThan(-10);
    expect(stats.alpha.mean).toBeLessThan(10);
    expect(stats.beta.mean).toBeGreaterThan(-15);
    expect(stats.beta.mean).toBeLessThan(15);
  }, 120000);

  test("model compiles to complete state", () => {
    const bound = wellsDistanceModel.bind(wellsDistanceData);
    expect(bound.state).toBe("complete");
    expect(bound.paramNames).toContain("alpha");
    expect(bound.paramNames).toContain("beta");
  });
});
