import { describe, test, expect, beforeAll } from "vitest";
import { init, numpy as np, grad } from "@jax-js/jax";
import { positive } from "../../src/constraints";

describe("positive constraint", () => {
  beforeAll(async () => {
    await init("cpu");
  });

  test("transform maps 0 to 1", () => {
    const c = positive();
    const x = np.array([0]);
    const y = c.transform(x);
    expect(y.item()).toBeCloseTo(1, 5);
  });

  test("transform maps -1 to exp(-1)", () => {
    const c = positive();
    const x = np.array([-1]);
    const y = c.transform(x);
    expect(y.item()).toBeCloseTo(Math.exp(-1), 5);
  });

  test("transform maps 1 to exp(1)", () => {
    const c = positive();
    const x = np.array([1]);
    const y = c.transform(x);
    expect(y.item()).toBeCloseTo(Math.exp(1), 5);
  });

  test("inverse is correct", () => {
    const c = positive();
    const x = np.array([0.5, 1.0, 2.0]);
    const y = c.transform(x.ref);
    const xBack = c.inverse(y);
    const xData = x.ref.dataSync();
    const xBackData = xBack.ref.dataSync();
    for (let i = 0; i < 3; i++) {
      expect(xBackData[i]).toBeCloseTo(xData[i], 5);
    }
  });

  test("transform/inverse roundtrip", () => {
    const c = positive();
    const constrained = np.array([0.5, 1.0, 2.0, 10.0]);
    const unconstrained = c.inverse(constrained.ref);
    const backToConstrained = c.transform(unconstrained);
    const original = constrained.ref.dataSync();
    const roundtrip = backToConstrained.ref.dataSync();
    for (let i = 0; i < 4; i++) {
      expect(roundtrip[i]).toBeCloseTo(original[i], 5);
    }
  });

  test("logDetJacobian equals sum of unconstrained values", () => {
    const c = positive();
    const x = np.array([0.5, 1.0, -0.5]);
    const ldj = c.logDetJacobian(x);
    // For exp transform: log|det J| = sum(x) for elementwise
    const expected = 0.5 + 1.0 + -0.5;
    expect(ldj.item()).toBeCloseTo(expected, 5);
  });

  test("logDetJacobian matches numerical differentiation", () => {
    const c = positive();
    const x = np.array([0.5]);

    // Analytical: log|d/dx exp(x)| = log(exp(x)) = x
    const analytical = c.logDetJacobian(x.ref);

    // Numerical: log|det J| where J = d transform / d x
    // For scalar, this is just log|d exp(x) / dx| = log|exp(x)| = x
    const h = 1e-4;
    const xPlus = np.array([0.5 + h]);
    const xMinus = np.array([0.5 - h]);
    const fPlus = c.transform(xPlus);
    const fMinus = c.transform(xMinus);
    const numericalDeriv = fPlus.sub(fMinus).div(2 * h).item();
    const numericalLdj = Math.log(Math.abs(numericalDeriv));

    expect(analytical.item()).toBeCloseTo(numericalLdj, 3);
  });
});
