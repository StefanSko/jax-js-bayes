import { describe, expect, test } from "vitest";
import { numpy as np, random } from "@jax-js/jax";
import { normal } from "../../src/distributions/normal";
import { meanValue, stdValue } from "../utils";

describe("normal distribution", () => {
  test("logProb matches analytical formula", () => {
    const d = normal(0, 1);
    const logp = d.logProb(np.array([0])).item();
    expect(logp).toBeCloseTo(-0.9189, 4);
  });

  test("samples have correct mean and std", () => {
    const d = normal(2, 3);
    const samples = d.sample(random.key(0), [10000]);
    expect(Math.abs(meanValue(samples) - 2)).toBeLessThan(0.1);
    expect(Math.abs(stdValue(samples) - 3)).toBeLessThan(0.1);
  });
});
