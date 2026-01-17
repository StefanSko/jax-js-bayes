import { beforeAll, describe, expect, test } from "vitest";
import { init, numpy as np, random } from "@jax-js/jax";
import { normal } from "../../src/distributions/normal";

beforeAll(async () => {
  await init();
});

describe("normal", () => {
  test("logProb matches analytical formula", () => {
    const dist = normal(0, 1);
    const lp = dist.logProb(np.array(0)).item();
    expect(lp).toBeCloseTo(-0.9189, 3);
  });

  test("samples have correct mean and std", () => {
    const dist = normal(2, 3);
    const draws = dist.sample(random.key(0), [5000]);
    const mean = np.mean(draws).item();
    const std = np.std(draws).item();
    expect(mean).toBeCloseTo(2, 1);
    expect(std).toBeCloseTo(3, 1);
  });
});
