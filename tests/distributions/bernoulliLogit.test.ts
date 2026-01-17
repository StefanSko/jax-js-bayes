import { describe, expect, test } from "vitest";
import { numpy as np } from "@jax-js/jax";
import { bernoulliLogit } from "../../src/distributions/bernoulliLogit";

describe("bernoulliLogit distribution", () => {
  test("logProb matches analytical formula", () => {
    const d = bernoulliLogit(0);
    const logp = d.logProb(np.array([1, 0])).item();
    const expected = Math.log(0.5) + Math.log(0.5);
    expect(logp).toBeCloseTo(expected, 4);
  });
});
