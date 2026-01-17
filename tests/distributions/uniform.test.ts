import { describe, test, expect, beforeAll } from "vitest";
import { init, numpy as np, random } from "@jax-js/jax";
import { uniform } from "../../src/distributions";

describe("uniform distribution", () => {
  beforeAll(async () => {
    await init("cpu");
  });

  test("logProb for Uniform(0, 1)", () => {
    const d = uniform(0, 1);
    const x = np.array([0.5]);
    const logP = d.logProb(x);
    // f(x) = 1, log(1) = 0
    expect(logP.item()).toBeCloseTo(0, 5);
  });

  test("logProb for Uniform(0, 2)", () => {
    const d = uniform(0, 2);
    const x = np.array([1]);
    const logP = d.logProb(x);
    // f(x) = 0.5, log(0.5) ≈ -0.693
    expect(logP.item()).toBeCloseTo(Math.log(0.5), 3);
  });

  test("logProb for Uniform(-1, 1)", () => {
    const d = uniform(-1, 1);
    const x = np.array([0]);
    const logP = d.logProb(x);
    // range = 2, f(x) = 0.5, log(0.5) ≈ -0.693
    expect(logP.item()).toBeCloseTo(Math.log(0.5), 3);
  });

  test("samples are within bounds", () => {
    const d = uniform(2, 5);
    const samples = d.sample(random.key(42), [1000]);
    const data = samples.ref.dataSync();
    for (const v of data) {
      expect(v).toBeGreaterThanOrEqual(2);
      expect(v).toBeLessThan(5);
    }
  });

  test("samples have correct mean (approximately)", () => {
    // Mean of Uniform(low, high) = (low + high) / 2
    const low = 2;
    const high = 6;
    const d = uniform(low, high);
    const samples = d.sample(random.key(42), [10000]);
    const mean = samples.sum().div(10000).item();
    expect(mean).toBeCloseTo((low + high) / 2, 1);
  });
});
