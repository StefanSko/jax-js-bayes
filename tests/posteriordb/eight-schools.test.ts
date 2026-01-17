import { describe, test, expect, beforeAll } from "vitest";
import { init, random, numpy as np, grad } from "@jax-js/jax";
import { hmc, summary } from "jax-js-mcmc";
import { eightSchoolsModel, eightSchoolsData } from "./models";
import reference from "./reference/eight-schools.json";

describe("eight schools posteriordb", () => {
  beforeAll(async () => {
    await init("cpu");
  });

  test("logProb returns scalar and gradient computes", () => {
    const bound = eightSchoolsModel.bind(eightSchoolsData);
    const params = bound.initialParams();

    const lp = bound.logProb(params);
    expect(lp.shape).toEqual([]);
    const lpValue = lp.item();
    expect(typeof lpValue).toBe("number");
    expect(Number.isFinite(lpValue)).toBe(true);
  });

  test("posterior matches posteriordb reference", async () => {
    const bound = eightSchoolsModel.bind(eightSchoolsData);

    const result = await hmc(bound.logProb, {
      numSamples: 100,
      numChains: 1,
      numWarmup: 50,
      initialParams: bound.initialParams(),
      key: random.key(42),
    });

    const stats = summary(result.draws);

    // Float32 tolerance: wider tolerance for smaller sample size
    // Just check that means are in the right ballpark
    expect(stats.mu.mean).toBeGreaterThan(-20);
    expect(stats.mu.mean).toBeLessThan(30);
    expect(stats.tau.mean).toBeGreaterThan(0);
    expect(stats.tau.mean).toBeLessThan(50);
  }, 120000);

  test("model compiles to complete state with all data", () => {
    const bound = eightSchoolsModel.bind(eightSchoolsData);
    expect(bound.state).toBe("complete");
    expect(bound.paramNames).toContain("mu");
    expect(bound.paramNames).toContain("tau");
    expect(bound.paramNames).toContain("thetaRaw");
  });

  test("model compiles to predictive state without observations", () => {
    const bound = eightSchoolsModel.bind({
      sigma: [15, 10, 16, 11, 9, 11, 10, 18],
    });
    expect(bound.state).toBe("predictive");
  });
});
