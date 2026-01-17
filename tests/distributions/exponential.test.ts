import { describe, test, expect, beforeAll } from "vitest";
import { init, numpy as np, random } from "@jax-js/jax";
import { exponential } from "../../src/distributions";

describe("exponential distribution", () => {
  beforeAll(async () => {
    await init("cpu");
  });

  test("logProb at x=0 for rate=1", () => {
    const d = exponential(1);
    const x = np.array([0]);
    const logP = d.logProb(x);
    // f(0) = 1*exp(-0) = 1, log(1) = 0
    expect(logP.item()).toBeCloseTo(0, 5);
  });

  test("logProb at x=1 for rate=1", () => {
    const d = exponential(1);
    const x = np.array([1]);
    const logP = d.logProb(x);
    // f(1) = exp(-1), log(exp(-1)) = -1
    expect(logP.item()).toBeCloseTo(-1, 5);
  });

  test("logProb with rate=2", () => {
    const d = exponential(2);
    const x = np.array([0]);
    const logP = d.logProb(x);
    // f(0) = 2*exp(0) = 2, log(2) ≈ 0.693
    expect(logP.item()).toBeCloseTo(Math.log(2), 3);
  });

  test("samples are positive", () => {
    const d = exponential(1);
    const samples = d.sample(random.key(42), [1000]);
    const data = samples.ref.dataSync();
    for (const v of data) {
      expect(v).toBeGreaterThanOrEqual(0);
    }
  });

  test("samples have correct mean (approximately)", () => {
    // Mean of Exponential(rate) = 1/rate
    const rate = 2;
    const d = exponential(rate);
    const samples = d.sample(random.key(42), [10000]);
    const mean = samples.sum().div(10000).item();
    expect(mean).toBeCloseTo(1 / rate, 1);
  });
});
