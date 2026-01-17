import { beforeAll, describe, expect, test } from "vitest";
import { init, numpy as np } from "@jax-js/jax";
import { bernoulliLogit } from "../../src/distributions/bernoulliLogit";

beforeAll(async () => {
  await init();
});

describe("bernoulliLogit", () => {
  test("logProb matches analytical formula", () => {
    const logit = Math.log(3);
    const dist = bernoulliLogit(logit);
    const lp1 = dist.logProb(np.array(1)).item();
    const lp0 = dist.logProb(np.array(0)).item();
    const p = 3 / 4;
    expect(lp1).toBeCloseTo(Math.log(p), 3);
    expect(lp0).toBeCloseTo(Math.log(1 - p), 3);
  });
});
