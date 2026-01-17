import { beforeAll, describe, expect, test } from "vitest";
import { init, numpy as np, random } from "@jax-js/jax";
import { exponential } from "../../src/distributions/exponential";

beforeAll(async () => {
  await init();
});

describe("exponential", () => {
  test("logProb matches analytical formula", () => {
    const dist = exponential(2);
    const lp = dist.logProb(np.array(1)).item();
    const expected = Math.log(2) - 2;
    expect(lp).toBeCloseTo(expected, 3);
  });

  test("samples have correct mean", () => {
    const dist = exponential(2);
    const draws = dist.sample(random.key(2), [5000]);
    const mean = np.mean(draws).item();
    expect(mean).toBeCloseTo(0.5, 1);
  });
});
