import { describe, expect, test } from "vitest";
import { numpy as np } from "@jax-js/jax";
import { bounded } from "../../src/constraints/bounded";
import { numericalLogDetJacobian } from "./utils";

describe("bounded constraint", () => {
  const c = bounded(-2, 3);

  test("transform maps R to bounds", () => {
    const values = c.transform(np.array([-10, 0, 10])).js() as number[];
    expect(values[0]).toBeGreaterThanOrEqual(-2);
    expect(values[2]).toBeLessThanOrEqual(3);
  });

  test("inverse is correct", () => {
    const x = np.array([-1.5, 0.0, 2.5]);
    const y = c.transform(x.ref);
    const roundtrip = c.inverse(y).js() as number[];
    const original = x.js() as number[];
    expect(roundtrip[0]).toBeCloseTo(original[0], 5);
    expect(roundtrip[1]).toBeCloseTo(original[1], 5);
    expect(roundtrip[2]).toBeCloseTo(original[2], 5);
  });

  test("logDetJacobian matches numerical derivative", () => {
    const x = 0.4;
    const analytical = c.logDetJacobian(np.array([x])).item();
    const numerical = numericalLogDetJacobian(c.transform, x);
    expect(analytical).toBeCloseTo(numerical, 3);
  });
});
