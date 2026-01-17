import { beforeAll, describe, expect, test } from "vitest";
import { init, numpy as np, random } from "@jax-js/jax";
import { halfNormal } from "../../src/distributions/halfNormal";

beforeAll(async () => {
  await init();
});

describe("halfNormal", () => {
  test("logProb matches base normal plus log(2)", () => {
    const dist = halfNormal(1);
    const lp = dist.logProb(np.array(1)).item();
    const expected = -0.5 * 1 - 0.5 * Math.log(2 * Math.PI) + Math.log(2);
    expect(lp).toBeCloseTo(expected, 3);
  });

  test("samples have correct mean", () => {
    const dist = halfNormal(2);
    const draws = dist.sample(random.key(1), [5000]);
    const mean = np.mean(draws).item();
    const expected = 2 * Math.sqrt(2 / Math.PI);
    expect(mean).toBeCloseTo(expected, 1);
  });
});
