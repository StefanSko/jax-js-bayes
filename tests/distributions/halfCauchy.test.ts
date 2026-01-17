import { beforeAll, describe, expect, test } from "vitest";
import { init, numpy as np } from "@jax-js/jax";
import { halfCauchy } from "../../src/distributions/halfCauchy";

beforeAll(async () => {
  await init();
});

describe("halfCauchy", () => {
  test("logProb matches analytical formula", () => {
    const dist = halfCauchy(1);
    const lp = dist.logProb(np.array(1)).item();
    const expected = -Math.log(Math.PI) - Math.log(1) - Math.log(1 + 1) + Math.log(2);
    expect(lp).toBeCloseTo(expected, 3);
  });
});
