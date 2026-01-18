import { describe, expect, test } from "vitest";
import { numpy as np } from "@jax-js/jax";
import { model, param, observed } from "../src/model";
import { normal, halfNormal } from "../src/distributions";
import { positive } from "../src/constraints";

describe("model DSL", () => {
  test("logProb returns scalar array", () => {
    const simple = model({
      mu: param(normal(0, 1)),
      sigma: param(halfNormal(1), { constraint: positive() }),
      y: observed(({ mu, sigma }) => normal(mu, sigma)),
    });

    const bound = simple.bind({ y: np.array([0, 1]) });
    const logp = bound.logProb({ mu: np.array(0), sigma: np.array(0) });
    expect(logp.shape).toEqual([]);
  });

  test("predictive binding omits observations", () => {
    const simple = model({
      mu: param(normal(0, 1)),
      sigma: param(halfNormal(1), { constraint: positive() }),
      y: observed(({ mu, sigma }) => normal(mu, sigma)),
    });

    const bound = simple.bind({});
    expect(bound.kind).toBe("predictive");
    expect("logProb" in bound).toBe(false);
  });

  test("bind rejects partial observed data", () => {
    const simple = model({
      mu: param(normal(0, 1)),
      y: observed(({ mu }) => normal(mu, 1)),
      z: observed(({ mu }) => normal(mu, 1)),
    });

    expect(() => simple.bind({ y: np.array([0]) })).toThrow(
      /partial observed/i,
    );
  });
});
