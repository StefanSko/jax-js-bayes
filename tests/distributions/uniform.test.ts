import { describe, test, expect } from "vitest";
import { numpy as np, random } from "@jax-js/jax";
import { uniform } from "../../src/distributions/uniform";

describe("uniform distribution", () => {
  test("logProb matches analytical formula for unit interval", () => {
    const d = uniform(0, 1);
    const x = 0.5;
    // Uniform logProb = -log(high - low) = -log(1) = 0
    const expected = 0;
    const result = d.logProb(np.array([x]));
    expect(result.item()).toBeCloseTo(expected, 4);
  });

  test("logProb matches analytical formula for non-unit interval", () => {
    const low = 2;
    const high = 5;
    const d = uniform(low, high);
    const x = 3;
    // logProb = -log(high - low) = -log(3)
    const expected = -Math.log(high - low);
    const result = d.logProb(np.array([x]));
    expect(result.item()).toBeCloseTo(expected, 4);
  });

  test("logProb sums over batched inputs", () => {
    const d = uniform(0, 2);
    const x = np.array([0.5, 1.0, 1.5]);
    const result = d.logProb(x);
    // Each point: -log(2)
    expect(result.item()).toBeCloseTo(3 * -Math.log(2), 4);
  });

  test("samples are within bounds", () => {
    const low = 2;
    const high = 5;
    const d = uniform(low, high);
    const key = random.key(42);
    const samples = d.sample(key, [1000]);
    const minVal = np.min(samples.ref).item();
    const maxVal = np.max(samples).item();
    expect(minVal).toBeGreaterThanOrEqual(low);
    expect(maxVal).toBeLessThan(high);
  });

  test("samples have correct mean", () => {
    const low = 2;
    const high = 5;
    const d = uniform(low, high);
    const key = random.key(42);
    const samples = d.sample(key, [10000]);
    // Mean of uniform is (low + high) / 2
    const expectedMean = (low + high) / 2;
    const mean = np.mean(samples).item();
    expect(mean).toBeCloseTo(expectedMean, 1);
  });

  test("support is bounded", () => {
    const low = 2;
    const high = 5;
    const d = uniform(low, high);
    expect(d.support).toEqual({ type: "bounded", low, high });
  });
});
