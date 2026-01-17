import { beforeAll, describe, expect, test } from "vitest";
import { init, numpy as np } from "@jax-js/jax";
import { bounded } from "../../src/constraints/bounded";

beforeAll(async () => {
  await init();
});

function numericalLogDetJacobian(low: number, high: number, x: number, eps = 1e-4) {
  const c = bounded(low, high);
  const fPlus = c.transform(np.array(x + eps)).item();
  const fMinus = c.transform(np.array(x - eps)).item();
  const deriv = (fPlus - fMinus) / (2 * eps);
  return Math.log(Math.abs(deriv));
}

describe("bounded constraint", () => {
  const c = bounded(0, 2);

  test("transform/inverse roundtrip", () => {
    const x = np.array(0.2);
    const y = c.transform(x.ref);
    const back = c.inverse(y).item();
    expect(back).toBeCloseTo(0.2, 4);
  });

  test("logDetJacobian matches numerical", () => {
    const x = 0.1;
    const analytical = c.logDetJacobian(np.array(x)).item();
    const numerical = numericalLogDetJacobian(0, 2, x);
    expect(analytical).toBeCloseTo(numerical, 3);
  });
});
