import { describe, test, expect } from "vitest";
import { numpy as np, random } from "@jax-js/jax";
import { halfNormal } from "../../src/distributions/halfNormal";

describe("halfNormal distribution", () => {
  test("logProb matches analytical formula", () => {
    const scale = 1;
    const d = halfNormal(scale);
    // Half-normal logProb at x=0: log(2) + normal(0, scale).logProb(0)
    // = log(2) - 0.5 * log(2π) ≈ log(2) - 0.9189
    const expected = Math.log(2) - 0.5 * Math.log(2 * Math.PI);
    const result = d.logProb(np.array([0]));
    expect(result.item()).toBeCloseTo(expected, 4);
  });

  test("logProb at positive x matches formula", () => {
    const scale = 2;
    const d = halfNormal(scale);
    const x = 1;
    // logProb = log(2) - 0.5 * (x/scale)^2 - log(scale) - 0.5 * log(2π)
    const z = x / scale;
    const expected = Math.log(2) - 0.5 * z * z - Math.log(scale) - 0.5 * Math.log(2 * Math.PI);
    const result = d.logProb(np.array([x]));
    expect(result.item()).toBeCloseTo(expected, 4);
  });

  test("logProb sums over batched inputs", () => {
    const d = halfNormal(1);
    const x = np.array([0, 0, 0]);
    const result = d.logProb(x);
    const singleLogProb = Math.log(2) - 0.5 * Math.log(2 * Math.PI);
    expect(result.item()).toBeCloseTo(3 * singleLogProb, 4);
  });

  test("samples are all positive", () => {
    const d = halfNormal(2);
    const key = random.key(42);
    const samples = d.sample(key, [1000]);
    const minVal = np.min(samples).item();
    expect(minVal).toBeGreaterThanOrEqual(0);
  });

  test("samples have correct mean for half-normal", () => {
    const scale = 2;
    const d = halfNormal(scale);
    const key = random.key(42);
    const samples = d.sample(key, [10000]);
    // Mean of half-normal is scale * sqrt(2/π)
    const expectedMean = scale * Math.sqrt(2 / Math.PI);
    const mean = np.mean(samples).item();
    expect(mean).toBeCloseTo(expectedMean, 0);
  });

  test("support is positive", () => {
    const d = halfNormal(1);
    expect(d.support).toEqual({ type: "positive" });
  });
});
