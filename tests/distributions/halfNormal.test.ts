import { describe, test, expect, beforeAll } from "vitest";
import { init, numpy as np, random } from "@jax-js/jax";
import { halfNormal } from "../../src/distributions";

describe("halfNormal distribution", () => {
  beforeAll(async () => {
    await init("cpu");
  });

  test("logProb at x=0 for scale=1", () => {
    const d = halfNormal(1);
    const x = np.array([0]);
    const logP = d.logProb(x);
    // f(0) = 2/sqrt(2π), log(2/sqrt(2π)) ≈ 0.2257
    expect(logP.item()).toBeCloseTo(Math.log(2 / Math.sqrt(2 * Math.PI)), 3);
  });

  test("logProb at x=1 for scale=1", () => {
    const d = halfNormal(1);
    const x = np.array([1]);
    const logP = d.logProb(x);
    // log(2/sqrt(2π)) - 0.5*1^2 ≈ -0.2743
    const expected = Math.log(2 / Math.sqrt(2 * Math.PI)) - 0.5;
    expect(logP.item()).toBeCloseTo(expected, 3);
  });

  test("logProb with scale=2", () => {
    const d = halfNormal(2);
    const x = np.array([0]);
    const logP = d.logProb(x);
    // log(2/sqrt(2π)) - log(2) ≈ -0.4672
    const expected = Math.log(2 / Math.sqrt(2 * Math.PI)) - Math.log(2);
    expect(logP.item()).toBeCloseTo(expected, 3);
  });

  test("samples are positive", () => {
    const d = halfNormal(1);
    const samples = d.sample(random.key(42), [1000]);
    const data = samples.ref.dataSync();
    for (const v of data) {
      expect(v).toBeGreaterThanOrEqual(0);
    }
  });

  test("samples have correct mean (approximately)", () => {
    // Mean of HalfNormal(scale) = scale * sqrt(2/π)
    const scale = 2;
    const d = halfNormal(scale);
    const samples = d.sample(random.key(42), [10000]);
    const mean = samples.sum().div(10000).item();
    const expectedMean = scale * Math.sqrt(2 / Math.PI);
    expect(mean).toBeCloseTo(expectedMean, 0);
  });
});
