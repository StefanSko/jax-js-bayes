import { beforeAll, describe, expect, test } from "vitest";
import { init, numpy as np, random } from "@jax-js/jax";
import { bernoulli } from "../../src/distributions/bernoulli";

beforeAll(async () => {
  await init();
});

describe("bernoulli", () => {
  test("logProb matches analytical formula", () => {
    const dist = bernoulli(0.8);
    const lp1 = dist.logProb(np.array(1)).item();
    const lp0 = dist.logProb(np.array(0)).item();
    expect(lp1).toBeCloseTo(Math.log(0.8), 3);
    expect(lp0).toBeCloseTo(Math.log(0.2), 3);
  });

  test("samples have correct mean", () => {
    const dist = bernoulli(0.3);
    const draws = dist.sample(random.key(4), [5000]);
    const mean = np.mean(draws).item();
    expect(mean).toBeCloseTo(0.3, 1);
  });
});
