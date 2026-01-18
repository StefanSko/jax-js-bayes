import { describe, test, expect } from "vitest";
import { numpy as np, random } from "@jax-js/jax";
import { halfCauchy } from "../../src/distributions/halfCauchy";

describe("halfCauchy distribution", () => {
  test("logProb matches analytical formula at x=0", () => {
    const scale = 1;
    const d = halfCauchy(scale);
    // Half-Cauchy logProb at x=0: log(2) + Cauchy(0, scale).logProb(0)
    // Cauchy logProb at 0 = -log(π) - log(scale)
    // Half-Cauchy at 0 = log(2) - log(π) - log(scale)
    const expected = Math.log(2) - Math.log(Math.PI) - Math.log(scale);
    const result = d.logProb(np.array([0]));
    expect(result.item()).toBeCloseTo(expected, 4);
  });

  test("logProb at positive x matches formula", () => {
    const scale = 2;
    const d = halfCauchy(scale);
    const x = 1;
    // Half-Cauchy logProb = log(2) - log(π) - log(scale) - log(1 + (x/scale)^2)
    const z = x / scale;
    const expected = Math.log(2) - Math.log(Math.PI) - Math.log(scale) - Math.log(1 + z * z);
    const result = d.logProb(np.array([x]));
    expect(result.item()).toBeCloseTo(expected, 4);
  });

  test("logProb sums over batched inputs", () => {
    const d = halfCauchy(1);
    const x = np.array([0, 0, 0]);
    const result = d.logProb(x);
    const singleLogProb = Math.log(2) - Math.log(Math.PI);
    expect(result.item()).toBeCloseTo(3 * singleLogProb, 4);
  });

  test("samples are all positive", () => {
    const d = halfCauchy(2);
    const key = random.key(42);
    const samples = d.sample(key, [1000]);
    const minVal = np.min(samples).item();
    expect(minVal).toBeGreaterThanOrEqual(0);
  });

  test("samples have heavy tails (some very large values)", () => {
    const scale = 1;
    const d = halfCauchy(scale);
    const key = random.key(42);
    const samples = d.sample(key, [10000]);
    // Cauchy has such heavy tails that max should be much larger than mean
    const maxVal = np.max(samples.ref).item();
    const mean = np.mean(samples).item();
    // Max should be at least 10x mean for Cauchy
    expect(maxVal).toBeGreaterThan(mean * 10);
  });

  test("support is positive", () => {
    const d = halfCauchy(1);
    expect(d.support).toEqual({ type: "positive" });
  });
});
