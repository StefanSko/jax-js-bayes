import { numpy as np } from "@jax-js/jax";
import { model, param, data, observed } from "../../src/model";
import {
  normal,
  halfCauchy,
  halfNormal,
  bernoulliLogit,
} from "../../src/distributions";
import { positive } from "../../src/constraints";

export const eightSchoolsNoncentered = model({
  mu: param(normal(0, 5)),
  tau: param(halfCauchy(5), { constraint: positive() }),
  thetaRaw: param(normal(0, 1), { shape: "school" }),
  theta: ({ mu, tau, thetaRaw }) => mu.add(tau.mul(thetaRaw)),
  sigma: data({ shape: "school" }),
  y: observed(({ theta, sigma }) => normal(theta, sigma)),
});

export const eightSchoolsCentered = model({
  mu: param(normal(0, 5)),
  tau: param(halfCauchy(5), { constraint: positive() }),
  theta: param(({ mu, tau }) => normal(mu, tau), { shape: "school" }),
  sigma: data({ shape: "school" }),
  y: observed(({ theta, sigma }) => normal(theta, sigma)),
});

export const kidscoreMomiq = model({
  beta0: param(normal(0, 100)),
  beta1: param(normal(0, 100)),
  sigma: param(halfCauchy(2.5), { constraint: positive() }),
  mom_iq: data(),
  mu: ({ beta0, beta1, mom_iq }) => beta0.add(beta1.mul(mom_iq)),
  kid_score: observed(({ mu, sigma }) => normal(mu, sigma)),
});

export const kidscoreInteraction = model({
  beta0: param(normal(0, 100)),
  beta1: param(normal(0, 100)),
  beta2: param(normal(0, 100)),
  beta3: param(normal(0, 100)),
  sigma: param(halfCauchy(2.5), { constraint: positive() }),
  mom_hs: data(),
  mom_iq: data(),
  mu: ({ beta0, beta1, beta2, beta3, mom_hs, mom_iq }) =>
    beta0
      .add(beta1.mul(mom_hs))
      .add(beta2.mul(mom_iq))
      .add(beta3.mul(mom_hs.mul(mom_iq))),
  kid_score: observed(({ mu, sigma }) => normal(mu, sigma)),
});

export const radonPooled = model({
  alpha: param(normal(0, 10)),
  beta: param(normal(0, 10)),
  sigma_y: param(halfNormal(1), { constraint: positive() }),
  floor_measure: data(),
  mu: ({ alpha, beta, floor_measure }) => alpha.add(beta.mul(floor_measure)),
  log_radon: observed(({ mu, sigma_y }) => normal(mu, sigma_y)),
});

export const radonHierarchical = model({
  mu_alpha: param(normal(0, 10)),
  sigma_alpha: param(halfNormal(1), { constraint: positive() }),
  sigma_y: param(halfNormal(1), { constraint: positive() }),
  J: data({ shape: "J" }),
  alpha_raw: param(normal(0, 1), { shape: "J" }),
  beta_uppm: param(normal(0, 10)),
  beta_floor: param(normal(0, 10)),
  alpha: ({ mu_alpha, sigma_alpha, alpha_raw }) => mu_alpha.add(sigma_alpha.mul(alpha_raw)),
  county_idx: data(),
  log_uppm: data(),
  floor_measure: data(),
  mu: ({ alpha, county_idx, log_uppm, floor_measure, beta_uppm, beta_floor }) => {
    const alphaByCounty = np.take(alpha, county_idx);
    return alphaByCounty.add(log_uppm.mul(beta_uppm)).add(floor_measure.mul(beta_floor));
  },
  log_radon: observed(({ mu, sigma_y }) => normal(mu, sigma_y)),
});

export const wellsDist = model({
  beta0: param(normal(0, 100)),
  beta1: param(normal(0, 100)),
  dist: data(),
  logits: ({ beta0, beta1, dist }) => beta0.add(beta1.mul(dist)),
  switched: observed(({ logits }) => bernoulliLogit(logits)),
});

export const blr = model({
  beta: param(normal(0, 10), { shape: "D" }),
  sigma: param(halfNormal(10), { constraint: positive() }),
  X: data({ shape: ["N", "D"] }),
  mu: ({ X, beta }) => np.matmul(X, beta),
  y: observed(({ mu, sigma }) => normal(mu, sigma)),
});

export const logearnHeight = model({
  beta0: param(normal(0, 100)),
  beta1: param(normal(0, 100)),
  sigma: param(halfNormal(100), { constraint: positive() }),
  height: data(),
  mu: ({ beta0, beta1, height }) => beta0.add(beta1.mul(height)),
  log_earn: observed(({ mu, sigma }) => normal(mu, sigma)),
});

export const earnHeight = model({
  beta0: param(normal(0, 1e6)),
  beta1: param(normal(0, 1e6)),
  sigma: param(halfNormal(1e6), { constraint: positive() }),
  height: data(),
  mu: ({ beta0, beta1, height }) => beta0.add(beta1.mul(height)),
  earn: observed(({ mu, sigma }) => normal(mu, sigma)),
});

export const mesquiteLogVolume = model({
  K: data({ shape: "K" }),
  beta: param(normal(0, 1e6), { shape: "K" }),
  sigma: param(halfNormal(1e6), { constraint: positive() }),
  diam1: data(),
  diam2: data(),
  canopy_height: data(),
  log_canopy_volume: ({ diam1, diam2, canopy_height }) =>
    np.log(diam1.mul(diam2).mul(canopy_height)),
  mu: ({ beta, log_canopy_volume }) =>
    np.take(beta, 0).add(np.take(beta, 1).mul(log_canopy_volume)),
  log_weight: observed(({ mu, sigma }) => normal(mu, sigma)),
});
