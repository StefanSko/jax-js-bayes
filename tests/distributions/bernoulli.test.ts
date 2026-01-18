import { describe, test, expect } from "vitest";
import { numpy as np, random } from "@jax-js/jax";
import { bernoulli } from "../../src/distributions/bernoulli";

describe("bernoulli distribution", () => {
  test("logProb for x=1 matches analytical formula", () => {
    const p = 0.7;
    const d = bernoulli(p);
    // logProb(1) = log(p)
    const expected = Math.log(p);
    const result = d.logProb(np.array([1]));
    expect(result.item()).toBeCloseTo(expected, 4);
  });

  test("logProb for x=0 matches analytical formula", () => {
    const p = 0.7;
    const d = bernoulli(p);
    // logProb(0) = log(1 - p)
    const expected = Math.log(1 - p);
    const result = d.logProb(np.array([0]));
    expect(result.item()).toBeCloseTo(expected, 4);
  });

  test("logProb sums over batched inputs", () => {
    const p = 0.5;
    const d = bernoulli(p);
    const x = np.array([1, 0, 1]);
    const result = d.logProb(x);
    // 2 successes + 1 failure: 2*log(0.5) + 1*log(0.5) = 3*log(0.5)
    expect(result.item()).toBeCloseTo(3 * Math.log(0.5), 4);
  });

  test("samples are 0 or 1", () => {
    const d = bernoulli(0.5);
    const key = random.key(42);
    const samples = d.sample(key, [1000]);
    const minVal = np.min(samples.ref).item();
    const maxVal = np.max(samples).item();
    // Samples should be boolean/0-1
    expect(minVal).toBeGreaterThanOrEqual(0);
    expect(maxVal).toBeLessThanOrEqual(1);
  });

  test("samples have correct mean", () => {
    const p = 0.3;
    const d = bernoulli(p);
    const key = random.key(42);
    const samples = d.sample(key, [10000]);
    const mean = np.mean(samples).item();
    expect(mean).toBeCloseTo(p, 1);
  });

  test("support is discrete", () => {
    const d = bernoulli(0.5);
    expect(d.support).toEqual({ type: "discrete" });
  });
});
