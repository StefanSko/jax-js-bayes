import { model, param, observed, data, normal, halfNormal, halfCauchy, positive, take, bernoulliLogit } from "../../src";
import { numpy as np, type Array } from "@jax-js/jax";

// Eight Schools (Non-centered parameterization)
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

// Eight Schools (Centered parameterization)
export const eightSchoolsCenteredModel = model({
  mu: param(normal(0, 5)),
  tau: param(halfCauchy(5), { constraint: positive() }),
  theta: param(normal(0, 1), { shape: [8] }),
  // theta directly parameterized with prior N(mu, tau)
  // But we need to evaluate prior differently - use reparameterization in likelihood
  thetaScaled: ({ mu, tau, theta }: Record<string, Array>) =>
    mu.ref.add(tau.ref.mul(theta)),
  sigma: data({ shape: [8] }),
  y: observed(({ thetaScaled, sigma }: Record<string, Array>) => normal(thetaScaled, sigma)),
});

// Kidscore: Linear regression of kid test scores on mom IQ
export const kidscoreModel = model({
  alpha: param(normal(0, 100)),
  beta: param(normal(0, 10)),
  sigma: param(halfNormal(50), { constraint: positive() }),
  mu: ({ alpha, beta, mom_iq }: Record<string, Array>) =>
    alpha.ref.add(beta.ref.mul(mom_iq)),
  mom_iq: data(),
  kid_score: observed(({ mu, sigma }: Record<string, Array>) => normal(mu, sigma)),
});

export const kidscoreData = {
  mom_iq: [121.12, 89.36, 115.44, 99.45, 92.75, 107.90, 138.89, 106.77, 125.15, 81.62,
           95.07, 74.00, 113.33, 100.77, 91.82, 95.07, 82.51, 104.12, 108.02, 83.84],
  kid_score: [65, 83, 108, 115, 98, 69, 123, 79, 119, 88,
              73, 63, 90, 96, 89, 92, 59, 93, 84, 61],
};

// Radon Pooled: Simple pooled regression (no hierarchy)
export const radonPooledModel = model({
  alpha: param(normal(0, 10)),
  beta: param(normal(0, 10)),
  sigma: param(halfNormal(5), { constraint: positive() }),
  mu: ({ alpha, beta, floor }: Record<string, Array>) =>
    alpha.ref.add(beta.ref.mul(floor)),
  floor: data(),
  log_radon: observed(({ mu, sigma }: Record<string, Array>) => normal(mu, sigma)),
});

// Radon Pooled data (subset - 20 observations)
export const radonPooledData = {
  floor: [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0],
  log_radon: [0.83, 0.99, 1.10, 1.63, 0.64, 1.55, 1.63, 1.48, 0.29, 1.58,
              1.31, 0.59, 1.95, 1.57, 1.89, 0.59, 0.34, 1.68, 1.97, 1.51],
};

// Radon Hierarchical: Varying intercepts by county
export const radonHierarchicalModel = model({
  mu_alpha: param(normal(0, 10)),
  sigma_alpha: param(halfNormal(5), { constraint: positive() }),
  alphaRaw: param(normal(0, 1), { shape: [5] }), // 5 counties in subset
  beta: param(normal(0, 10)),
  sigma: param(halfNormal(5), { constraint: positive() }),
  // County-level intercepts (non-centered)
  alpha: ({ mu_alpha, sigma_alpha, alphaRaw }: Record<string, Array>) =>
    mu_alpha.ref.add(sigma_alpha.ref.mul(alphaRaw)),
  // Index into county-level parameters
  mu: ({ alpha, beta, floor, county }: Record<string, Array>) =>
    take(alpha.ref, county).add(beta.ref.mul(floor)),
  floor: data(),
  county: data(),
  log_radon: observed(({ mu, sigma }: Record<string, Array>) => normal(mu, sigma)),
});

// Radon Hierarchical data (subset - 20 observations, 5 counties)
export const radonHierarchicalData = {
  floor: [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0],
  county: np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4], { dtype: "int32" }),
  log_radon: [0.83, 0.99, 1.10, 1.63, 0.64, 1.55, 1.63, 1.48, 0.29, 1.58,
              1.31, 0.59, 1.95, 1.57, 1.89, 0.59, 0.34, 1.68, 1.97, 1.51],
};

// Wells Distance: Logistic regression for well switching
export const wellsDistanceModel = model({
  alpha: param(normal(0, 10)),
  beta: param(normal(0, 5)),
  // Linear predictor
  eta: ({ alpha, beta, dist100 }: Record<string, Array>) =>
    alpha.ref.add(beta.ref.mul(dist100)),
  dist100: data(),
  switched: observed(({ eta }: Record<string, Array>) => bernoulliLogit(eta)),
});

// Wells Distance data (subset - 20 observations)
export const wellsDistanceData = {
  dist100: [0.35, 0.64, 0.89, 0.24, 0.47, 0.12, 0.73, 0.56, 0.91, 0.18,
            0.42, 0.68, 0.33, 0.77, 0.21, 0.55, 0.84, 0.29, 0.61, 0.46],
  switched: [1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1],
};
