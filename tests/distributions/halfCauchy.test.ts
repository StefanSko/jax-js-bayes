import { describe, expect, test } from "vitest";
import { numpy as np, random } from "@jax-js/jax";
import { halfCauchy } from "../../src/distributions/halfCauchy";

describe("halfCauchy distribution", () => {
  test("logProb matches analytical formula", () => {
    const d = halfCauchy(2);
    const x = 1;
    const logp = d.logProb(np.array([x])).item();
    const expected = Math.log(2) - Math.log(Math.PI * 2 * (1 + (x / 2) ** 2));
    expect(logp).toBeCloseTo(expected, 4);
  });

  test("samples are non-negative", () => {
    const d = halfCauchy(1);
    const samples = d.sample(random.key(2), [1000]);
    const min = np.min(samples).item();
    expect(min).toBeGreaterThanOrEqual(0);
  });
});
