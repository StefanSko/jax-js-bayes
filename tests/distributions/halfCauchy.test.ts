import { describe, test, expect, beforeAll } from "vitest";
import { init, numpy as np, random } from "@jax-js/jax";
import { halfCauchy } from "../../src/distributions";

describe("halfCauchy distribution", () => {
  beforeAll(async () => {
    await init("cpu");
  });

  test("logProb at x=0 for scale=1", () => {
    const d = halfCauchy(1);
    const x = np.array([0]);
    const logP = d.logProb(x);
    // f(0) = 2/π, log(2/π) ≈ -0.4515
    expect(logP.item()).toBeCloseTo(Math.log(2 / Math.PI), 3);
  });

  test("logProb at x=1 for scale=1", () => {
    const d = halfCauchy(1);
    const x = np.array([1]);
    const logP = d.logProb(x);
    // f(1) = 2/(π * (1 + 1)) = 1/π, log(1/π) ≈ -1.1447
    expect(logP.item()).toBeCloseTo(Math.log(1 / Math.PI), 3);
  });

  test("logProb with scale=5", () => {
    const d = halfCauchy(5);
    const x = np.array([0]);
    const logP = d.logProb(x);
    // f(0) = 2/(π*5), log(2/(5π)) ≈ -2.0604
    expect(logP.item()).toBeCloseTo(Math.log(2 / (5 * Math.PI)), 3);
  });

  test("logProb handles batched input", () => {
    const d = halfCauchy(1);
    const x = np.array([0, 1, 2, 3]);
    const logP = d.logProb(x);
    expect(logP.shape).toEqual([4]);
  });

  test("samples are positive", () => {
    const d = halfCauchy(1);
    const samples = d.sample(random.key(42), [1000]);
    const data = samples.ref.dataSync();
    for (const v of data) {
      expect(v).toBeGreaterThanOrEqual(0);
    }
  });

  test("samples have correct median (approximately)", () => {
    // Median of HalfCauchy(scale) is scale
    const d = halfCauchy(5);
    const samples = d.sample(random.key(42), [10000]);
    const data = [...samples.ref.dataSync()].sort((a, b) => a - b);
    const median = data[5000];
    // Median should be close to scale (5)
    expect(median).toBeGreaterThan(3);
    expect(median).toBeLessThan(8);
  });
});
