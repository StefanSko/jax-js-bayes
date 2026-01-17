import { beforeAll, describe, expect, test } from "vitest";
import { init, numpy as np } from "@jax-js/jax";
import { model, param, observed, data } from "../src/model";
import { normal } from "../src/distributions/normal";

beforeAll(async () => {
  await init();
});

describe("model", () => {
  test("binds complete data and computes logProb", () => {
    const m = model({
      mu: param(normal(0, 1)),
      sigma: data({}),
      y: observed(({ mu, sigma }) => normal(mu, sigma)),
    });

    const bound = m.bind({
      sigma: [1, 1],
      y: [1, 2],
    });

    const mu = 0.5;
    const lp = bound.logProb({ mu }).item();

    const logNorm = (x: number, mean: number, sd: number) =>
      -0.5 * Math.pow((x - mean) / sd, 2) - Math.log(sd) - 0.5 * Math.log(2 * Math.PI);
    const expected =
      logNorm(mu, 0, 1) +
      logNorm(1, mu, 1) +
      logNorm(2, mu, 1);

    expect(lp).toBeCloseTo(expected, 3);
  });

  test("predictive binding omits observations", () => {
    const m = model({
      mu: param(normal(0, 1)),
      sigma: data({}),
      y: observed(({ mu, sigma }) => normal(mu, sigma)),
    });

    const bound = m.bind({
      sigma: [1, 1],
    });

    expect(bound.state).toBe("predictive");
  });

  test("rejects partial observations", () => {
    const m = model({
      mu: param(normal(0, 1)),
      sigma: data({}),
      y: observed(({ mu, sigma }) => normal(mu, sigma)),
      z: observed(({ mu }) => normal(mu, 1)),
    });

    expect(() =>
      m.bind({
        sigma: [1, 1],
        y: [1, 2],
      }),
    ).toThrow(/provide all observed/i);
  });
});
