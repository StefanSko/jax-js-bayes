import { describe, expect, test } from "vitest";
import { numpy as np, random } from "@jax-js/jax";
import { exponential } from "../../src/distributions/exponential";
import { meanValue } from "../utils";

describe("exponential distribution", () => {
  test("logProb matches analytical formula", () => {
    const d = exponential(2);
    const x = 1;
    const logp = d.logProb(np.array([x])).item();
    const expected = Math.log(2) - 2 * x;
    expect(logp).toBeCloseTo(expected, 4);
  });

  test("samples have correct mean", () => {
    const rate = 3;
    const d = exponential(rate);
    const samples = d.sample(random.key(3), [20000]);
    const mean = meanValue(samples);
    expect(Math.abs(mean - 1 / rate)).toBeLessThan(0.05);
  });
});
