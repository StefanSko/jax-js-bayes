import { describe, test, expect, beforeAll } from "vitest";
import { init, random, numpy as np } from "@jax-js/jax";
import { model, param, observed, data } from "../../src/model";
import { normal } from "../../src/distributions/normal";
import { halfCauchy } from "../../src/distributions/halfCauchy";
import { positive } from "../../src/constraints/positive";

describe("samplePrior", () => {
  beforeAll(async () => {
    await init("cpu");
  });

  test("samples from prior distributions", () => {
    const testModel = model({
      mu: param(normal(0, 5)),
      sigma: param(halfCauchy(2), { constraint: positive() }),
      y: observed(({ mu, sigma }) => normal(mu, sigma)),
    });

    const bound = testModel.bind({ y: [1, 2, 3] });
    const samples = bound.samplePrior(random.key(42));

    // Check that samples were generated for all params
    expect("mu" in samples).toBe(true);
    expect("sigma" in samples).toBe(true);

    // mu should be in unconstrained space (normal sample)
    const mu = (samples as Record<string, any>).mu;
    expect(mu.shape).toEqual([]);
    expect(typeof mu.item()).toBe("number");

    // sigma should be in unconstrained space (log of positive value)
    const sigma = (samples as Record<string, any>).sigma;
    expect(sigma.shape).toEqual([]);
    expect(typeof sigma.item()).toBe("number");
  });

  test("samples correct shapes for array parameters", () => {
    const testModel = model({
      theta: param(normal(0, 1), { shape: [8] }),
      y: observed(({ theta }) => normal(theta, np.array(1))),
    });

    const bound = testModel.bind({ y: [1, 2, 3, 4, 5, 6, 7, 8] });
    const samples = bound.samplePrior(random.key(42));

    const theta = (samples as Record<string, any>).theta;
    expect(theta.shape).toEqual([8]);
  });

  test("different keys produce different samples", () => {
    const testModel = model({
      mu: param(normal(0, 5)),
      y: observed(({ mu }) => normal(mu, np.array(1))),
    });

    const bound = testModel.bind({ y: [1] });
    const samples1 = bound.samplePrior(random.key(42));
    const samples2 = bound.samplePrior(random.key(43));

    const mu1 = (samples1 as Record<string, any>).mu.item();
    const mu2 = (samples2 as Record<string, any>).mu.item();

    expect(mu1).not.toEqual(mu2);
  });
});

describe("simulate", () => {
  beforeAll(async () => {
    await init("cpu");
  });

  test("generates simulated observations", () => {
    const testModel = model({
      mu: param(normal(0, 5)),
      sigma: param(halfCauchy(2), { constraint: positive() }),
      y: observed(({ mu, sigma }) => normal(mu, sigma)),
    });

    const bound = testModel.bind({ y: [1, 2, 3] });

    // Create unconstrained params (sigma in log space)
    const params = {
      mu: np.array(2.0),
      sigma: np.array(0.0), // log(1) = 0, so sigma = 1
    };

    const simulated = bound.simulate(params, random.key(42));

    expect("y" in simulated).toBe(true);
    expect(simulated.y.shape).toEqual([3]);
  });

  test("respects parameter values", () => {
    const testModel = model({
      mu: param(normal(0, 5)),
      y: observed(({ mu }) => normal(mu, np.array(0.01))), // Very small sigma
    });

    const bound = testModel.bind({ y: [0, 0, 0, 0, 0] });

    // Set mu to 100
    const params = { mu: np.array(100.0) };

    const simulated = bound.simulate(params, random.key(42));

    // All samples should be close to 100
    const yValues = [...simulated.y.dataSync()] as number[];
    for (const val of yValues) {
      expect(val).toBeGreaterThan(99);
      expect(val).toBeLessThan(101);
    }
  });

  test("different keys produce different samples", () => {
    const testModel = model({
      mu: param(normal(0, 5)),
      y: observed(({ mu }) => normal(mu, np.array(1))),
    });

    const bound = testModel.bind({ y: [1, 2, 3] });
    const params = { mu: np.array(0.0) };

    const sim1 = bound.simulate(params, random.key(42));
    const sim2 = bound.simulate(params, random.key(43));

    const y1 = [...sim1.y.dataSync()] as number[];
    const y2 = [...sim2.y.dataSync()] as number[];

    // At least one element should differ
    const anyDifferent = y1.some((v, i) => v !== y2[i]);
    expect(anyDifferent).toBe(true);
  });

  test("works with derived quantities", () => {
    const testModel = model({
      mu: param(normal(0, 5)),
      tau: param(halfCauchy(5), { constraint: positive() }),
      thetaRaw: param(normal(0, 1), { shape: [3] }),
      // Derived: theta = mu + tau * thetaRaw
      theta: ({ mu, tau, thetaRaw }) => mu.add(tau.mul(thetaRaw)),
      sigma: data(),
      y: observed(({ theta, sigma }) => normal(theta, sigma)),
    });

    const bound = testModel.bind({
      sigma: [1, 1, 1],
      y: [0, 0, 0],
    });

    const params = {
      mu: np.array(5.0),
      tau: np.array(0.0), // log(1) = 0, so tau = 1
      thetaRaw: np.array([0, 0, 0]),
    };

    const simulated = bound.simulate(params, random.key(42));

    expect("y" in simulated).toBe(true);
    expect(simulated.y.shape).toEqual([3]);

    // With thetaRaw = [0,0,0] and tau = 1, theta = [5,5,5]
    // So y should be centered around 5
    const yValues = [...simulated.y.dataSync()] as number[];
    const mean = yValues.reduce((a, b) => a + b, 0) / yValues.length;
    // Mean should be roughly 5 (with some noise)
    expect(mean).toBeGreaterThan(2);
    expect(mean).toBeLessThan(8);
  });
});
