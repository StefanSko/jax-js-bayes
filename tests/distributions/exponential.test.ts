import { describe, test, expect } from "vitest";
import { numpy as np, random } from "@jax-js/jax";
import { exponential } from "../../src/distributions/exponential";

describe("exponential distribution", () => {
  test("logProb matches analytical formula", () => {
    const rate = 2;
    const d = exponential(rate);
    const x = 1;
    // Exponential logProb = log(rate) - rate * x
    const expected = Math.log(rate) - rate * x;
    const result = d.logProb(np.array([x]));
    expect(result.item()).toBeCloseTo(expected, 4);
  });

  test("logProb at x=0", () => {
    const rate = 2;
    const d = exponential(rate);
    // logProb at 0 = log(rate)
    const expected = Math.log(rate);
    const result = d.logProb(np.array([0]));
    expect(result.item()).toBeCloseTo(expected, 4);
  });

  test("logProb sums over batched inputs", () => {
    const rate = 1;
    const d = exponential(rate);
    const x = np.array([1, 1, 1]);
    const result = d.logProb(x);
    // Each point: log(1) - 1*1 = -1
    expect(result.item()).toBeCloseTo(-3, 4);
  });

  test("samples are all positive", () => {
    const d = exponential(1);
    const key = random.key(42);
    const samples = d.sample(key, [1000]);
    const minVal = np.min(samples).item();
    expect(minVal).toBeGreaterThanOrEqual(0);
  });

  test("samples have correct mean", () => {
    const rate = 2;
    const d = exponential(rate);
    const key = random.key(42);
    const samples = d.sample(key, [10000]);
    // Mean of exponential is 1/rate
    const expectedMean = 1 / rate;
    const mean = np.mean(samples).item();
    expect(mean).toBeCloseTo(expectedMean, 1);
  });

  test("support is positive", () => {
    const d = exponential(1);
    expect(d.support).toEqual({ type: "positive" });
  });
});
