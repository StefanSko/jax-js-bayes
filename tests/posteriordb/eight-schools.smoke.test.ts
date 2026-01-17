import { describe, expect, test } from "vitest";
import { Array as JaxArray, numpy as np, random } from "@jax-js/jax";
import { hmc } from "jax-js-mcmc";

import { model, param, observed, data } from "../../src/index";
import { normal, halfCauchy } from "../../src/distributions/index";
import { positive } from "../../src/constraints/index";

const REF_MEANS = {
  mu: 4.41051833695493,
  tau: 3.60205952364059,
};

const TOLERANCE = {
  abs: 3.0,
  rel: 0.6,
};

const eightSchools = model({
  mu: param(normal(0, 5)),
  tau: param(halfCauchy(5), { constraint: positive() }),
  thetaRaw: param(normal(0, 1), { shape: [8] }),
  sigma: data({ shape: [8] }),
  y: observed(({ mu, tau, thetaRaw, sigma }) =>
    normal(mu.add(tau.mul(thetaRaw)), sigma),
  ),
});

const eightSchoolsData = {
  y: np.array([28, 8, -3, 7, -1, 1, 18, 12]),
  sigma: np.array([15, 10, 16, 11, 9, 11, 10, 18]),
};

const initialParams = {
  mu: np.array(0),
  tau: np.array(0),
  thetaRaw: np.zeros([8]),
};

function meanFromDraws(draws: JaxArray): number {
  const values = draws.js() as number[] | number[][];
  let sum = 0;
  let count = 0;

  if (Array.isArray(values[0])) {
    for (const row of values as number[][]) {
      for (const value of row) {
        sum += value;
        count += 1;
      }
    }
  } else {
    for (const value of values as number[]) {
      sum += value;
      count += 1;
    }
  }

  if (count === 0) {
    throw new Error("Expected non-empty draws");
  }

  return sum / count;
}

function expectMeanClose(label: string, value: number, ref: number): void {
  const allowed = Math.max(TOLERANCE.abs, Math.abs(ref) * TOLERANCE.rel);
  const delta = Math.abs(value - ref);

  expect(Number.isFinite(value)).toBe(true);
  if (delta > allowed) {
    throw new Error(
      `${label} mean off by ${delta} (allowed ${allowed}, ref ${ref})`,
    );
  }
}

describe("posteriordb smoke", () => {
  test("eight schools noncentered matches reference means", async () => {
    const bound = eightSchools.bind(eightSchoolsData);
    const result = await hmc(bound.logProb, {
      numSamples: 100,
      numWarmup: 100,
      numChains: 1,
      key: random.key(0),
      initialParams,
    });

    const muMean = meanFromDraws(result.draws.mu);
    const tauMean = meanFromDraws(result.draws.tau);

    expectMeanClose("mu", muMean, REF_MEANS.mu);
    expectMeanClose("tau", tauMean, REF_MEANS.tau);
  });
});
