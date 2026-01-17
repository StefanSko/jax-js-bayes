import { describe, expect, test } from "vitest";
import { numpy as np } from "@jax-js/jax";
import { positive } from "../../src/constraints/positive";
import { numericalLogDetJacobian } from "./utils";

describe("positive constraint", () => {
  const c = positive();

  test("transform maps R to R+", () => {
    const values = c.transform(np.array([-1, 0, 1])).js() as number[];
    expect(values[0]).toBeGreaterThan(0);
    expect(values[1]).toBeCloseTo(1, 6);
  });

  test("inverse is correct", () => {
    const x = np.array([0.5, 1.0, 2.0]);
    const y = c.transform(x.ref);
    const roundtrip = c.inverse(y).js() as number[];
    const original = x.js() as number[];
    expect(roundtrip[0]).toBeCloseTo(original[0], 5);
    expect(roundtrip[1]).toBeCloseTo(original[1], 5);
    expect(roundtrip[2]).toBeCloseTo(original[2], 5);
  });

  test("logDetJacobian matches numerical derivative", () => {
    const x = 0.5;
    const analytical = c.logDetJacobian(np.array([x])).item();
    const numerical = numericalLogDetJacobian(c.transform, x);
    expect(analytical).toBeCloseTo(numerical, 4);
  });
});
