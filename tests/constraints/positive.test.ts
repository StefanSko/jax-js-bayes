import { beforeAll, describe, expect, test } from "vitest";
import { init, numpy as np } from "@jax-js/jax";
import { positive } from "../../src/constraints/positive";

beforeAll(async () => {
  await init();
});

function numericalLogDetJacobian(x: number, eps = 1e-4) {
  const c = positive();
  const fPlus = c.transform(np.array(x + eps)).item();
  const fMinus = c.transform(np.array(x - eps)).item();
  const deriv = (fPlus - fMinus) / (2 * eps);
  return Math.log(Math.abs(deriv));
}

describe("positive constraint", () => {
  const c = positive();

  test("transform/inverse roundtrip", () => {
    const x = np.array(0.5);
    const y = c.transform(x.ref);
    const back = c.inverse(y).item();
    expect(back).toBeCloseTo(0.5, 4);
  });

  test("logDetJacobian matches numerical", () => {
    const x = 0.3;
    const analytical = c.logDetJacobian(np.array(x)).item();
    const numerical = numericalLogDetJacobian(x);
    expect(analytical).toBeCloseTo(numerical, 3);
  });
});
