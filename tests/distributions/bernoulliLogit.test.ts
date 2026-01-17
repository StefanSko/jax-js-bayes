import { describe, test, expect } from "vitest";
import { numpy as np, random } from "@jax-js/jax";
import { bernoulliLogit } from "../../src/distributions/bernoulliLogit";

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

describe("bernoulliLogit distribution", () => {
  test("logProb for x=1 matches analytical formula", () => {
    const logit = 1;
    const d = bernoulliLogit(logit);
    // logProb(1) = -log(1 + exp(-logit)) = logit - log(1 + exp(logit))
    // Using log-sum-exp stable form: -softplus(-logit)
    const expected = -Math.log(1 + Math.exp(-logit));
    const result = d.logProb(np.array([1]));
    expect(result.item()).toBeCloseTo(expected, 4);
  });

  test("logProb for x=0 matches analytical formula", () => {
    const logit = 1;
    const d = bernoulliLogit(logit);
    // logProb(0) = log(1 - sigmoid(logit)) = -logit - log(1 + exp(-logit))
    // = -softplus(logit)
    const expected = -Math.log(1 + Math.exp(logit));
    const result = d.logProb(np.array([0]));
    expect(result.item()).toBeCloseTo(expected, 4);
  });

  test("logProb matches bernoulli(sigmoid(logit))", () => {
    const logit = 0.5;
    const p = sigmoid(logit);
    const d = bernoulliLogit(logit);

    // Test x=1
    const result1 = d.logProb(np.array([1]));
    expect(result1.item()).toBeCloseTo(Math.log(p), 4);

    // Test x=0
    const result0 = d.logProb(np.array([0]));
    expect(result0.item()).toBeCloseTo(Math.log(1 - p), 4);
  });

  test("logProb sums over batched inputs", () => {
    const logit = 0;
    const d = bernoulliLogit(logit);
    const x = np.array([1, 0, 1]);
    const result = d.logProb(x);
    // At logit=0, p=0.5, so logProb is same for 0 and 1
    // Each is -log(2), total is -3*log(2)
    expect(result.item()).toBeCloseTo(-3 * Math.log(2), 4);
  });

  test("samples are 0 or 1", () => {
    const d = bernoulliLogit(0);
    const key = random.key(42);
    const samples = d.sample(key, [1000]);
    const minVal = np.min(samples.ref).item();
    const maxVal = np.max(samples).item();
    expect(minVal).toBeGreaterThanOrEqual(0);
    expect(maxVal).toBeLessThanOrEqual(1);
  });

  test("samples have correct mean", () => {
    const logit = 1;
    const d = bernoulliLogit(logit);
    const key = random.key(42);
    const samples = d.sample(key, [10000]);
    const expectedMean = sigmoid(logit);
    const mean = np.mean(samples).item();
    expect(mean).toBeCloseTo(expectedMean, 1);
  });

  test("support is discrete", () => {
    const d = bernoulliLogit(0);
    expect(d.support).toEqual({ type: "discrete" });
  });

  test("works with Array parameter", () => {
    const logits = np.array([0, 1, -1]);
    const d = bernoulliLogit(logits);
    // Test with all ones
    const x = np.array([1, 1, 1]);
    const result = d.logProb(x);
    // Sum of log(sigmoid(logit)) for each
    const expected =
      -Math.log(1 + Math.exp(0)) +
      -Math.log(1 + Math.exp(-1)) +
      -Math.log(1 + Math.exp(1));
    expect(result.item()).toBeCloseTo(expected, 4);
  });
});
