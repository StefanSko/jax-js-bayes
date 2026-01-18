import { describe, test, expect } from "vitest";
import { numpy as np } from "@jax-js/jax";
import { bounded } from "../../src/constraints/bounded";

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

function logit(p: number): number {
  return Math.log(p / (1 - p));
}

describe("bounded constraint", () => {
  test("transform maps R to (low, high)", () => {
    const low = 0;
    const high = 1;
    const c = bounded(low, high);

    // At x=0, sigmoid(0) = 0.5, so transform(0) = 0 + (1-0)*0.5 = 0.5
    expect(c.transform(np.array([0])).item()).toBeCloseTo(0.5, 4);

    // Large positive -> close to high
    const largePos = c.transform(np.array([10])).item();
    expect(largePos).toBeGreaterThan(0.99);
    expect(largePos).toBeLessThan(1);

    // Large negative -> close to low
    const largeNeg = c.transform(np.array([-10])).item();
    expect(largeNeg).toBeGreaterThan(0);
    expect(largeNeg).toBeLessThan(0.01);
  });

  test("transform works with non-unit bounds", () => {
    const low = 2;
    const high = 5;
    const c = bounded(low, high);

    // At x=0, transform = low + (high - low) * sigmoid(0) = 2 + 3*0.5 = 3.5
    expect(c.transform(np.array([0])).item()).toBeCloseTo(3.5, 4);
  });

  test("inverse is correct", () => {
    const c = bounded(0, 1);
    // Test individual values
    for (const xVal of [-2, 0, 2]) {
      const x = np.array([xVal]);
      const y = c.transform(x.ref);
      const xRecovered = c.inverse(y);
      expect(xRecovered.item()).toBeCloseTo(x.item(), 3);
    }
  });

  test("transform(inverse(y)) = y roundtrip", () => {
    const c = bounded(2, 5);
    // Test individual values
    for (const yVal of [2.5, 3.5, 4.5]) {
      const y = np.array([yVal]);
      const x = c.inverse(y.ref);
      const yRecovered = c.transform(x);
      expect(yRecovered.item()).toBeCloseTo(y.item(), 3);
    }
  });

  test("logDetJacobian at x=0 is correct", () => {
    const low = 0;
    const high = 1;
    const range = high - low;
    const c = bounded(low, high);

    // For sigmoid transform: d/dx (range * sigmoid(x)) = range * sigmoid(x) * (1 - sigmoid(x))
    // log|det(J)| = log(range) + log(sigmoid(x)) + log(1 - sigmoid(x))
    const x = 0;
    const s = sigmoid(x);
    const expected = Math.log(range) + Math.log(s) + Math.log(1 - s);
    const result = c.logDetJacobian(np.array([x])).item();
    expect(result).toBeCloseTo(expected, 4);
  });

  test("logDetJacobian sums over batch", () => {
    const c = bounded(0, 1);
    const x = np.array([0, 0, 0]);
    const result = c.logDetJacobian(x).item();
    // Each is log(0.5) + log(0.5) = 2*log(0.5), so 3 of them is 6*log(0.5)
    const singleValue = Math.log(0.5) + Math.log(0.5);
    expect(result).toBeCloseTo(3 * singleValue, 4);
  });

  test("logDetJacobian matches numerical differentiation", () => {
    const c = bounded(0, 1);
    const x = 0.5;
    const eps = 1e-4;

    const fPlus = c.transform(np.array([x + eps])).item();
    const fMinus = c.transform(np.array([x - eps])).item();
    const numericalDerivative = (fPlus - fMinus) / (2 * eps);
    const numericalLogDetJ = Math.log(numericalDerivative);

    const analytical = c.logDetJacobian(np.array([x])).item();
    expect(analytical).toBeCloseTo(numericalLogDetJ, 2);
  });
});
