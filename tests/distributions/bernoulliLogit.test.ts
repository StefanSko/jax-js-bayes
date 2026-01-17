import { describe, test, expect, beforeAll } from "vitest";
import { init, numpy as np, random } from "@jax-js/jax";
import { bernoulliLogit } from "../../src/distributions";

describe("bernoulliLogit distribution", () => {
  beforeAll(async () => {
    await init("cpu");
  });

  test("logProb for x=1, logit=0 (p=0.5)", () => {
    const d = bernoulliLogit(0);
    const x = np.array([1]);
    const logP = d.logProb(x);
    // P(x=1|logit=0) = sigmoid(0) = 0.5
    // log(0.5) ≈ -0.693
    expect(logP.item()).toBeCloseTo(Math.log(0.5), 3);
  });

  test("logProb for x=0, logit=0 (p=0.5)", () => {
    const d = bernoulliLogit(0);
    const x = np.array([0]);
    const logP = d.logProb(x);
    // P(x=0|logit=0) = 1 - sigmoid(0) = 0.5
    // log(0.5) ≈ -0.693
    expect(logP.item()).toBeCloseTo(Math.log(0.5), 3);
  });

  test("logProb for x=1, logit=2", () => {
    const d = bernoulliLogit(2);
    const x = np.array([1]);
    const logP = d.logProb(x);
    // P(x=1|logit=2) = sigmoid(2) ≈ 0.88
    // log(sigmoid(2)) = 2 - softplus(2)
    const sigmoid2 = 1 / (1 + Math.exp(-2));
    expect(logP.item()).toBeCloseTo(Math.log(sigmoid2), 2);
  });

  test("samples are binary", () => {
    const d = bernoulliLogit(0);
    const samples = d.sample(random.key(42), [1000]);
    const data = samples.ref.dataSync();
    for (const v of data) {
      expect(v === 0 || v === 1).toBe(true);
    }
  });

  test("samples have correct mean (approximately)", () => {
    // Mean = sigmoid(logit)
    const logit = 1;
    const d = bernoulliLogit(logit);
    const samples = d.sample(random.key(42), [10000]);
    const mean = samples.ref.sum().div(10000).item();
    const expectedMean = 1 / (1 + Math.exp(-logit));
    expect(mean).toBeCloseTo(expectedMean, 1);
  });
});
