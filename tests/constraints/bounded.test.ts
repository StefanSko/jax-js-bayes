import { describe, test, expect, beforeAll } from "vitest";
import { init, numpy as np } from "@jax-js/jax";
import { bounded } from "../../src/constraints";

describe("bounded constraint", () => {
  beforeAll(async () => {
    await init("cpu");
  });

  test("transform maps 0 to midpoint", () => {
    const c = bounded(0, 1);
    const x = np.array([0]);
    const y = c.transform(x);
    // sigmoid(0) = 0.5, scaled to [0,1] = 0.5
    expect(y.item()).toBeCloseTo(0.5, 5);
  });

  test("transform maps large positive to upper bound", () => {
    const c = bounded(0, 1);
    const x = np.array([10]);
    const y = c.transform(x);
    // sigmoid(10) ≈ 0.99995
    expect(y.item()).toBeCloseTo(1, 3);
  });

  test("transform maps large negative to lower bound", () => {
    const c = bounded(0, 1);
    const x = np.array([-10]);
    const y = c.transform(x);
    // sigmoid(-10) ≈ 0.00005
    expect(y.item()).toBeCloseTo(0, 3);
  });

  test("transform with custom bounds", () => {
    const c = bounded(2, 8);
    const x = np.array([0]);
    const y = c.transform(x);
    // sigmoid(0) = 0.5, scaled to [2,8] = 2 + 0.5*6 = 5
    expect(y.item()).toBeCloseTo(5, 5);
  });

  test("inverse is correct", () => {
    const c = bounded(0, 1);
    const x = np.array([-1, 0, 1, 2]);
    const y = c.transform(x.ref);
    const xBack = c.inverse(y);
    const xData = x.ref.dataSync();
    const xBackData = xBack.ref.dataSync();
    for (let i = 0; i < 4; i++) {
      expect(xBackData[i]).toBeCloseTo(xData[i], 4);
    }
  });

  test("transform/inverse roundtrip", () => {
    const c = bounded(0, 10);
    const constrained = np.array([1, 5, 9]);
    const unconstrained = c.inverse(constrained.ref);
    const backToConstrained = c.transform(unconstrained);
    const original = constrained.ref.dataSync();
    const roundtrip = backToConstrained.ref.dataSync();
    for (let i = 0; i < 3; i++) {
      expect(roundtrip[i]).toBeCloseTo(original[i], 4);
    }
  });

  test("logDetJacobian is finite at reasonable values", () => {
    const c = bounded(0, 1);
    const x = np.array([0, 1, -1]);
    const ldj = c.logDetJacobian(x);
    expect(Number.isFinite(ldj.item())).toBe(true);
  });
});
