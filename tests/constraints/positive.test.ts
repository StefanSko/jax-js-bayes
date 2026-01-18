import { describe, test, expect } from "vitest";
import { numpy as np, grad } from "@jax-js/jax";
import { positive } from "../../src/constraints/positive";

describe("positive constraint", () => {
  test("transform maps R to R+", () => {
    const c = positive();
    expect(c.transform(np.array([0])).item()).toBeCloseTo(1, 4);
    expect(c.transform(np.array([-1])).item()).toBeCloseTo(Math.exp(-1), 4);
    expect(c.transform(np.array([1])).item()).toBeCloseTo(Math.exp(1), 4);
  });

  test("inverse is correct", () => {
    const c = positive();
    // Test individual values
    expect(c.inverse(np.array([Math.exp(0.5)])).item()).toBeCloseTo(0.5, 4);
    expect(c.inverse(np.array([Math.exp(1.0)])).item()).toBeCloseTo(1.0, 4);
    expect(c.inverse(np.array([Math.exp(2.0)])).item()).toBeCloseTo(2.0, 4);
  });

  test("transform(inverse(y)) = y roundtrip", () => {
    const c = positive();
    // Test individual values
    for (const yVal of [0.5, 1.0, 2.0, 10.0]) {
      const y = np.array([yVal]);
      const x = c.inverse(y.ref);
      const yRecovered = c.transform(x);
      expect(yRecovered.item()).toBeCloseTo(y.item(), 4);
    }
  });

  test("logDetJacobian is correct analytically", () => {
    const c = positive();
    // For exp transform: d/dx exp(x) = exp(x), so log|det(J)| = x (for scalar)
    // For batched: sum of x values
    const x = np.array([0.5]);
    const analytical = c.logDetJacobian(x);
    expect(analytical.item()).toBeCloseTo(0.5, 4);
  });

  test("logDetJacobian sums over batch", () => {
    const c = positive();
    const x = np.array([1, 2, 3]);
    const analytical = c.logDetJacobian(x);
    // Sum should be 1 + 2 + 3 = 6
    expect(analytical.item()).toBeCloseTo(6, 4);
  });

  test("logDetJacobian matches numerical differentiation", () => {
    const c = positive();
    const x = np.array([0.5]);

    // Numerical log det Jacobian: log(d/dx transform(x))
    // For scalar, this is log(transform(x + eps) - transform(x - eps)) / (2*eps)
    const eps = 1e-4;
    const fPlus = c.transform(np.array([0.5 + eps])).item();
    const fMinus = c.transform(np.array([0.5 - eps])).item();
    const numericalDerivative = (fPlus - fMinus) / (2 * eps);
    const numericalLogDetJ = Math.log(numericalDerivative);

    const analytical = c.logDetJacobian(x).item();
    expect(analytical).toBeCloseTo(numericalLogDetJ, 3);
  });
});
