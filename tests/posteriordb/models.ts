import { model, param, observed, data, normal, halfNormal, halfCauchy, positive } from "../../src";
import type { Array } from "@jax-js/jax";

export const eightSchoolsModel = model({
  mu: param(normal(0, 5)),
  tau: param(halfCauchy(5), { constraint: positive() }),
  thetaRaw: param(normal(0, 1), { shape: [8] }),
  theta: ({ mu, tau, thetaRaw }: Record<string, Array>) =>
    mu.ref.add(tau.ref.mul(thetaRaw)),
  sigma: data({ shape: [8] }),
  y: observed(({ theta, sigma }: Record<string, Array>) => normal(theta, sigma)),
});

export const eightSchoolsData = {
  y: [28, 8, -3, 7, -1, 1, 18, 12],
  sigma: [15, 10, 16, 11, 9, 11, 10, 18],
};

// Kidscore: Linear regression of kid test scores on mom IQ
export const kidscoreModel = model({
  alpha: param(normal(0, 100)),
  beta: param(normal(0, 10)),
  sigma: param(halfNormal(50), { constraint: positive() }),
  // mu[i] = alpha + beta * mom_iq[i]
  mu: ({ alpha, beta, mom_iq }: Record<string, Array>) =>
    alpha.ref.add(beta.ref.mul(mom_iq)),
  mom_iq: data(),
  kid_score: observed(({ mu, sigma }: Record<string, Array>) => normal(mu, sigma)),
});

// Kidscore data (subset for testing)
export const kidscoreData = {
  mom_iq: [121.12, 89.36, 115.44, 99.45, 92.75, 107.90, 138.89, 106.77, 125.15, 81.62,
           95.07, 74.00, 113.33, 100.77, 91.82, 95.07, 82.51, 104.12, 108.02, 83.84],
  kid_score: [65, 83, 108, 115, 98, 69, 123, 79, 119, 88,
              73, 63, 90, 96, 89, 92, 59, 93, 84, 61],
};
