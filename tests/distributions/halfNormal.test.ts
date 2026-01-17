import { describe, expect, test } from "vitest";
import { numpy as np, random } from "@jax-js/jax";
import { halfNormal } from "../../src/distributions/halfNormal";
import { meanValue, stdValue } from "../utils";

describe("halfNormal distribution", () => {
  test("logProb matches analytical formula", () => {
    const d = halfNormal(1);
    const logp = d.logProb(np.array([1])).item();
    const expected = Math.log(2) - 0.5 * (1 * 1) - 0.5 * Math.log(2 * Math.PI);
    expect(logp).toBeCloseTo(expected, 4);
  });

  test("samples have correct mean and std", () => {
    const scale = 2;
    const d = halfNormal(scale);
    const samples = d.sample(random.key(1), [20000]);
    const meanExpected = scale * Math.sqrt(2 / Math.PI);
    const stdExpected = scale * Math.sqrt(1 - 2 / Math.PI);
    expect(Math.abs(meanValue(samples) - meanExpected)).toBeLessThan(0.1);
    expect(Math.abs(stdValue(samples) - stdExpected)).toBeLessThan(0.1);
  });
});
