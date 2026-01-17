import { describe, expect, test } from "vitest";
import { numpy as np, random } from "@jax-js/jax";
import { uniform } from "../../src/distributions/uniform";
import { meanValue } from "../utils";

describe("uniform distribution", () => {
  test("logProb matches analytical formula", () => {
    const d = uniform(0, 2);
    const logp = d.logProb(np.array([1])).item();
    const expected = -Math.log(2);
    expect(logp).toBeCloseTo(expected, 4);
  });

  test("samples have correct mean", () => {
    const d = uniform(-1, 3);
    const samples = d.sample(random.key(4), [20000]);
    const mean = meanValue(samples);
    expect(Math.abs(mean - 1)).toBeLessThan(0.05);
  });
});
