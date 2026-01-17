import { beforeAll, describe, expect, test } from "vitest";
import { init, numpy as np, random } from "@jax-js/jax";
import { uniform } from "../../src/distributions/uniform";

beforeAll(async () => {
  await init();
});

describe("uniform", () => {
  test("logProb matches analytical formula", () => {
    const dist = uniform(0, 2);
    const lp = dist.logProb(np.array(1)).item();
    const expected = -Math.log(2);
    expect(lp).toBeCloseTo(expected, 3);
  });

  test("samples have correct mean", () => {
    const dist = uniform(-1, 3);
    const draws = dist.sample(random.key(3), [5000]);
    const mean = np.mean(draws).item();
    const expected = 1;
    expect(mean).toBeCloseTo(expected, 1);
  });
});
