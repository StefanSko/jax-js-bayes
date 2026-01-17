import { describe, test, expect, beforeAll } from "vitest";
import { init, numpy as np, random } from "@jax-js/jax";
import { normal } from "../../src/distributions";

describe("normal distribution", () => {
  beforeAll(async () => {
    await init("cpu");
  });

  test("logProb matches analytical formula for standard normal", () => {
    const d = normal(0, 1);
    const x = np.array([0]);
    const logP = d.logProb(x);
    // -0.5 * log(2π) ≈ -0.9189
    expect(logP.item()).toBeCloseTo(-0.9189, 3);
  });

  test("logProb for x=1 in standard normal", () => {
    const d = normal(0, 1);
    const x = np.array([1]);
    const logP = d.logProb(x);
    // -0.5 * (1^2 + log(2π)) ≈ -1.4189
    expect(logP.item()).toBeCloseTo(-1.4189, 3);
  });

  test("logProb with non-zero mean", () => {
    const d = normal(2, 1);
    const x = np.array([2]);
    const logP = d.logProb(x);
    expect(logP.item()).toBeCloseTo(-0.9189, 3);
  });

  test("logProb with non-unit scale", () => {
    const d = normal(0, 2);
    const x = np.array([0]);
    const logP = d.logProb(x);
    // -0.5 * log(2π) - log(2) ≈ -1.6120
    expect(logP.item()).toBeCloseTo(-1.6120, 3);
  });

  test("logProb handles batched input", () => {
    const d = normal(0, 1);
    const x = np.array([0, 1, -1]);
    const logP = d.logProb(x);
    expect(logP.shape).toEqual([3]);
  });

  test("samples have correct mean", () => {
    const d = normal(2, 1);
    const samples = d.sample(random.key(42), [10000]);
    const mean = samples.sum().div(10000).item();
    expect(mean).toBeCloseTo(2, 1);
  });

  test("samples have correct std", () => {
    const d = normal(0, 3);
    const samples = d.sample(random.key(42), [10000]);
    const mean = samples.ref.sum().div(10000);
    const centered = samples.sub(mean);
    const variance = centered.ref.mul(centered).sum().div(10000 - 1);
    const std = np.sqrt(variance).item();
    expect(std).toBeCloseTo(3, 0);
  });
});
