import { describe, test, expect } from "vitest";
import { numpy as np, random } from "@jax-js/jax";
import { normal } from "../../src/distributions/normal";

describe("normal distribution", () => {
  test("logProb matches analytical formula for standard normal", () => {
    const d = normal(0, 1);
    // Standard normal logProb at x=0: -0.5 * log(2π) ≈ -0.9189
    const result = d.logProb(np.array([0]));
    const expected = -0.5 * Math.log(2 * Math.PI);
    expect(result.item()).toBeCloseTo(expected, 4);
  });

  test("logProb matches analytical formula for non-standard normal", () => {
    const loc = 2;
    const scale = 3;
    const d = normal(loc, scale);
    const x = 5;
    // logProb = -0.5 * ((x - loc) / scale)^2 - log(scale) - 0.5 * log(2π)
    const z = (x - loc) / scale;
    const expected = -0.5 * z * z - Math.log(scale) - 0.5 * Math.log(2 * Math.PI);
    const result = d.logProb(np.array([x]));
    expect(result.item()).toBeCloseTo(expected, 4);
  });

  test("logProb sums over batched inputs", () => {
    const d = normal(0, 1);
    const x = np.array([0, 0, 0]);
    const result = d.logProb(x);
    // Should be 3 times the single-point logProb
    const singleLogProb = -0.5 * Math.log(2 * Math.PI);
    expect(result.item()).toBeCloseTo(3 * singleLogProb, 4);
  });

  test("samples have correct mean and std", () => {
    const loc = 2;
    const scale = 3;
    const d = normal(loc, scale);
    const key = random.key(42);
    const samples = d.sample(key, [10000]);

    const mean = np.mean(samples.ref).item();
    const std = np.std(samples).item();

    expect(mean).toBeCloseTo(loc, 0);
    expect(std).toBeCloseTo(scale, 0);
  });

  test("support is real", () => {
    const d = normal(0, 1);
    expect(d.support).toEqual({ type: "real" });
  });

  test("works with Array parameters", () => {
    const loc = np.array([0, 1, 2]);
    const scale = np.array([1, 2, 3]);
    const d = normal(loc, scale);
    const x = np.array([0, 1, 2]);
    const result = d.logProb(x);
    // Each point is at the mean, so z = 0 for all
    // logProb = sum(-log(scale) - 0.5 * log(2π))
    const expected = -Math.log(1) - Math.log(2) - Math.log(3) - 3 * 0.5 * Math.log(2 * Math.PI);
    expect(result.item()).toBeCloseTo(expected, 4);
  });
});
