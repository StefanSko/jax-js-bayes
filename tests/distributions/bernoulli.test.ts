import { describe, expect, test } from "vitest";
import { numpy as np, random } from "@jax-js/jax";
import { bernoulli } from "../../src/distributions/bernoulli";
import { meanValue } from "../utils";

describe("bernoulli distribution", () => {
  test("logProb matches analytical formula", () => {
    const d = bernoulli(0.25);
    const logp = d.logProb(np.array([0, 1])).item();
    const expected = Math.log(1 - 0.25) + Math.log(0.25);
    expect(logp).toBeCloseTo(expected, 4);
  });

  test("samples have correct mean", () => {
    const p = 0.7;
    const d = bernoulli(p);
    const samples = d.sample(random.key(5), [20000]);
    const mean = meanValue(samples);
    expect(Math.abs(mean - p)).toBeLessThan(0.02);
  });
});
