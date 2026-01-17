import { describe, test, expect } from "vitest";
import { numpy as np, random, grad } from "@jax-js/jax";
import { model, param, data, observed } from "../src/model";
import { normal, halfNormal } from "../src/distributions";
import { positive } from "../src/constraints";

describe("model DSL", () => {
  describe("param helper", () => {
    test("creates param spec with prior", () => {
      const spec = param(normal(0, 1));
      expect(spec.type).toBe("param");
      expect(spec.prior).toBeDefined();
      expect(spec.constraint).toBeUndefined();
    });

    test("creates param spec with constraint", () => {
      const spec = param(halfNormal(1), { constraint: positive() });
      expect(spec.type).toBe("param");
      expect(spec.constraint).toBeDefined();
    });

    test("creates param spec with shape", () => {
      const spec = param(normal(0, 1), { shape: [3] });
      expect(spec.shape).toEqual([3]);
    });
  });

  describe("data helper", () => {
    test("creates data spec", () => {
      const spec = data();
      expect(spec.type).toBe("data");
    });

    test("creates data spec with shape", () => {
      const spec = data({ shape: "school" });
      expect(spec.type).toBe("data");
      expect(spec.shape).toBe("school");
    });
  });

  describe("observed helper", () => {
    test("creates observed spec", () => {
      const spec = observed(({ mu, sigma }) => normal(mu, sigma));
      expect(spec.type).toBe("observed");
      expect(typeof spec.likelihood).toBe("function");
    });
  });

  describe("simple model", () => {
    test("model can be defined", () => {
      const m = model({
        mu: param(normal(0, 10)),
        sigma: param(halfNormal(1), { constraint: positive() }),
        x: data(),
        y: observed(({ mu, sigma }) => normal(mu, sigma)),
      });

      expect(m.spec).toBeDefined();
    });

    test("model can be bound with complete data", () => {
      const m = model({
        mu: param(normal(0, 10)),
        sigma: param(halfNormal(1), { constraint: positive() }),
        y: observed(({ mu, sigma }) => normal(mu, sigma)),
      });

      const bound = m.bind({
        y: [1, 2, 3, 4, 5],
      });

      expect(bound.state).toBe("complete");
      expect(bound.logProb).toBeDefined();
    });

    test("bind throws when required data is missing", () => {
      const m = model({
        x: data(),
        y: observed(({ x }) => normal(x, 1)),
      });

      expect(() =>
        m.bind({
          y: [1, 2, 3],
        }),
      ).toThrow("Missing data");
    });

    test("predictive binding works with covariates only", () => {
      const m = model({
        x: data(),
        y: observed(({ x }) => normal(x, 1)),
      });

      const bound = m.bind({
        x: [1, 2, 3],
      });

      expect(bound.state).toBe("predictive");
    });

    test("logProb returns scalar Array", () => {
      const m = model({
        mu: param(normal(0, 10)),
        sigma: param(halfNormal(1), { constraint: positive() }),
        y: observed(({ mu, sigma }) => normal(mu, sigma)),
      });

      const bound = m.bind({
        y: [1, 2, 3],
      });

      // logProb takes unconstrained parameters
      // mu is unconstrained, sigma_raw is unconstrained (will be exp'd)
      const logP = bound.logProb({ mu: np.array(0), sigma: np.array(0) });
      expect(np.ndim(logP)).toBe(0); // scalar
      const logPValue = logP.item();
      expect(typeof logPValue).toBe("number");
      expect(Number.isFinite(logPValue)).toBe(true);
    });

    test("logProb is differentiable", () => {
      const m = model({
        mu: param(normal(0, 10)),
        sigma: param(halfNormal(1), { constraint: positive() }),
        y: observed(({ mu, sigma }) => normal(mu, sigma)),
      });

      const bound = m.bind({
        y: [1, 2, 3],
      });

      // Wrap logProb for grad - needs to work with flat parameter tree
      const logProbFn = (params: { mu: any; sigma: any }) => {
        return bound.logProb(params);
      };

      const gradLogProb = grad(logProbFn);
      const params = { mu: np.array(0), sigma: np.array(0) };
      const grads = gradLogProb(params);

      expect(grads.mu).toBeDefined();
      expect(grads.sigma).toBeDefined();
      expect(Number.isFinite(grads.mu.item())).toBe(true);
      expect(Number.isFinite(grads.sigma.item())).toBe(true);
    });
  });

  describe("model with derived quantities", () => {
    test("derived quantities are computed", () => {
      const m = model({
        mu: param(normal(0, 5)),
        tau: param(halfNormal(5), { constraint: positive() }),
        thetaRaw: param(normal(0, 1), { shape: [3] }),
        // Derived: theta = mu + tau * thetaRaw
        theta: ({ mu, tau, thetaRaw }) => mu.add(tau.mul(thetaRaw)),
        sigma: data({ shape: [3] }),
        y: observed(({ theta, sigma }) => normal(theta, sigma)),
      });

      const bound = m.bind({
        y: [28, 8, -3],
        sigma: [15, 10, 16],
      });

      expect(bound.state).toBe("complete");

      const params = {
        mu: np.array(5),
        tau: np.array(0),  // exp(0) = 1
        thetaRaw: np.array([0, 0, 0]),
      };

      const logP = bound.logProb(params);
      expect(Number.isFinite(logP.item())).toBe(true);
    });
  });

  describe("samplePrior", () => {
    test("samples parameters from prior", () => {
      const m = model({
        mu: param(normal(0, 10)),
        sigma: param(halfNormal(1), { constraint: positive() }),
        y: observed(({ mu, sigma }) => normal(mu, sigma)),
      });

      const key = random.key(42);
      const samples = m.samplePrior({ key });

      expect(samples.mu).toBeDefined();
      expect(samples.sigma).toBeDefined();
      expect(np.ndim(samples.mu)).toBe(0);
      expect(np.ndim(samples.sigma)).toBe(0);
      // sigma should be positive (from half-normal)
      expect(samples.sigma.item()).toBeGreaterThan(0);
    });

    test("samplePrior with shaped parameters", () => {
      const m = model({
        mu: param(normal(0, 5)),
        theta: param(normal(0, 1), { shape: "school" }),
        y: observed(({ mu }) => normal(mu, 1)),
      });

      const key = random.key(42);
      const samples = m.samplePrior({ key, dims: { school: 8 } });

      expect(samples.mu).toBeDefined();
      expect(samples.theta).toBeDefined();
      expect(np.shape(samples.theta)).toEqual([8]);
    });
  });

  describe("simulate", () => {
    test("simulates data from parameters", () => {
      const m = model({
        mu: param(normal(0, 10)),
        sigma: param(halfNormal(1), { constraint: positive() }),
        y: observed(({ mu, sigma }) => normal(mu, sigma)),
      });

      const params = { mu: np.array(5), sigma: np.array(2) };
      const key = random.key(42);
      const simulated = m.simulate(params, { key, dims: { y: 10 } });

      expect(simulated.y).toBeDefined();
      expect(np.shape(simulated.y)).toEqual([10]);
    });
  });
});
