import { describe, test } from "vitest";
import { numpy as np, random } from "@jax-js/jax";
import { hmc } from "jax-js-mcmc";
import { loadData, loadMeanStats } from "./pdb";
import { drawMean, expectMeanClose } from "./helpers";
import { disposeHmcResult, logMemory, maybeDisposeTree } from "./memory";
import {
  eightSchoolsNoncentered,
  eightSchoolsCentered,
  kidscoreMomiq,
  kidscoreInteraction,
  radonPooled,
  radonHierarchical,
  wellsDist,
  blr,
  logearnHeight,
  earnHeight,
  mesquiteLogVolume,
} from "./models";

const HMC_CONFIG = {
  numSamples: 400,
  numWarmup: 200,
  numChains: 2,
};

describe("posteriordb", () => {
  test("eight schools noncentered", async () => {
    logMemory("test start: eight schools noncentered");
    const eightStats = loadMeanStats("eight_schools-eight_schools_noncentered");
    const data = loadData("eight_schools");
    const bound = eightSchoolsNoncentered.bind({
      y: np.array(data.y as number[]),
      sigma: np.array(data.sigma as number[]),
    });
    const initialParams = {
      mu: np.array(0),
      tau: np.array(0),
      thetaRaw: np.zeros([data.J as number]),
    };

    const result = await hmc(bound.logProb, {
      ...HMC_CONFIG,
      key: random.key(0),
      initialParams,
    });

    const muMean = drawMean(result.draws.mu);
    const tauMean = drawMean(result.draws.tau);

    const muRef = eightStats.mean_value[eightStats.names.indexOf("mu")];
    const tauRef = eightStats.mean_value[eightStats.names.indexOf("tau")];

    expectMeanClose("mu", muMean, muRef);
    expectMeanClose("tau", tauMean, tauRef);
    disposeHmcResult(result);
    maybeDisposeTree(initialParams, "initialParams");
    logMemory("test end: eight schools noncentered");
  });

  test("eight schools centered", async () => {
    logMemory("test start: eight schools centered");
    const eightStats = loadMeanStats("eight_schools-eight_schools_noncentered");
    const data = loadData("eight_schools");
    const bound = eightSchoolsCentered.bind({
      y: np.array(data.y as number[]),
      sigma: np.array(data.sigma as number[]),
    });
    const initialParams = {
      mu: np.array(0),
      tau: np.array(0),
      theta: np.zeros([data.J as number]),
    };

    const result = await hmc(bound.logProb, {
      ...HMC_CONFIG,
      key: random.key(1),
      initialParams,
    });

    const muMean = drawMean(result.draws.mu);
    const tauMean = drawMean(result.draws.tau);

    const muRef = eightStats.mean_value[eightStats.names.indexOf("mu")];
    const tauRef = eightStats.mean_value[eightStats.names.indexOf("tau")];

    expectMeanClose("mu", muMean, muRef);
    expectMeanClose("tau", tauMean, tauRef);
    disposeHmcResult(result);
    maybeDisposeTree(initialParams, "initialParams");
    logMemory("test end: eight schools centered");
  });

  test("kidscore momiq", async () => {
    logMemory("test start: kidscore momiq");
    const kidiqStats = loadMeanStats("kidiq-kidscore_momiq");
    const data = loadData("kidiq");
    const bound = kidscoreMomiq.bind({
      mom_iq: np.array(data.mom_iq as number[]),
      kid_score: np.array(data.kid_score as number[]),
    });
    const initialParams = {
      beta0: np.array(0),
      beta1: np.array(0),
      sigma: np.array(0),
    };

    const result = await hmc(bound.logProb, {
      ...HMC_CONFIG,
      key: random.key(2),
      initialParams,
    });

    const beta0Mean = drawMean(result.draws.beta0);
    const beta1Mean = drawMean(result.draws.beta1);
    const sigmaMean = drawMean(result.draws.sigma);

    const beta0Ref = kidiqStats.mean_value[kidiqStats.names.indexOf("beta[1]")];
    const beta1Ref = kidiqStats.mean_value[kidiqStats.names.indexOf("beta[2]")];
    const sigmaRef = kidiqStats.mean_value[kidiqStats.names.indexOf("sigma")];

    expectMeanClose("beta0", beta0Mean, beta0Ref);
    expectMeanClose("beta1", beta1Mean, beta1Ref);
    expectMeanClose("sigma", sigmaMean, sigmaRef);
    disposeHmcResult(result);
    maybeDisposeTree(initialParams, "initialParams");
    logMemory("test end: kidscore momiq");
  });

  test("kidscore interaction", async () => {
    logMemory("test start: kidscore interaction");
    const kidiqInteractionStats = loadMeanStats("kidiq-kidscore_interaction");
    const data = loadData("kidiq");
    const bound = kidscoreInteraction.bind({
      mom_hs: np.array(data.mom_hs as number[]),
      mom_iq: np.array(data.mom_iq as number[]),
      kid_score: np.array(data.kid_score as number[]),
    });
    const initialParams = {
      beta0: np.array(0),
      beta1: np.array(0),
      beta2: np.array(0),
      beta3: np.array(0),
      sigma: np.array(0),
    };

    const result = await hmc(bound.logProb, {
      ...HMC_CONFIG,
      key: random.key(3),
      initialParams,
    });

    const beta0Mean = drawMean(result.draws.beta0);
    const beta1Mean = drawMean(result.draws.beta1);
    const beta2Mean = drawMean(result.draws.beta2);
    const beta3Mean = drawMean(result.draws.beta3);
    const sigmaMean = drawMean(result.draws.sigma);

    const beta0Ref =
      kidiqInteractionStats.mean_value[kidiqInteractionStats.names.indexOf("beta[1]")];
    const beta1Ref =
      kidiqInteractionStats.mean_value[kidiqInteractionStats.names.indexOf("beta[2]")];
    const beta2Ref =
      kidiqInteractionStats.mean_value[kidiqInteractionStats.names.indexOf("beta[3]")];
    const beta3Ref =
      kidiqInteractionStats.mean_value[kidiqInteractionStats.names.indexOf("beta[4]")];
    const sigmaRef =
      kidiqInteractionStats.mean_value[kidiqInteractionStats.names.indexOf("sigma")];

    expectMeanClose("beta0", beta0Mean, beta0Ref);
    expectMeanClose("beta1", beta1Mean, beta1Ref);
    expectMeanClose("beta2", beta2Mean, beta2Ref);
    expectMeanClose("beta3", beta3Mean, beta3Ref);
    expectMeanClose("sigma", sigmaMean, sigmaRef);
    disposeHmcResult(result);
    maybeDisposeTree(initialParams, "initialParams");
    logMemory("test end: kidscore interaction");
  });

  test("blr", async () => {
    logMemory("test start: blr");
    const blrStats = loadMeanStats("sblrc-blr");
    const data = loadData("sblrc");
    const bound = blr.bind({
      X: np.array(data.X as number[][]),
      y: np.array(data.y as number[]),
    });
    const initialParams = {
      beta: np.zeros([data.D as number]),
      sigma: np.array(0),
    };

    const result = await hmc(bound.logProb, {
      ...HMC_CONFIG,
      key: random.key(4),
      initialParams,
    });

    const beta1Mean = drawMean(np.take(result.draws.beta.ref, 0, -1));
    const sigmaMean = drawMean(result.draws.sigma);

    const beta1Ref = blrStats.mean_value[blrStats.names.indexOf("beta[1]")];
    const sigmaRef = blrStats.mean_value[blrStats.names.indexOf("sigma")];

    expectMeanClose("beta[1]", beta1Mean, beta1Ref);
    expectMeanClose("sigma", sigmaMean, sigmaRef);
    disposeHmcResult(result);
    maybeDisposeTree(initialParams, "initialParams");
    logMemory("test end: blr");
  });

  test("logearn height", async () => {
    logMemory("test start: logearn height");
    const logearnStats = loadMeanStats("earnings-logearn_height");
    const data = loadData("earnings");
    const bound = logearnHeight.bind({
      height: np.array(data.height as number[]),
      log_earn: np.log(np.array(data.earn as number[])),
    });
    const initialParams = {
      beta0: np.array(0),
      beta1: np.array(0),
      sigma: np.array(0),
    };

    const result = await hmc(bound.logProb, {
      ...HMC_CONFIG,
      key: random.key(5),
      initialParams,
    });

    const beta0Mean = drawMean(result.draws.beta0);
    const beta1Mean = drawMean(result.draws.beta1);
    const sigmaMean = drawMean(result.draws.sigma);

    const beta0Ref = logearnStats.mean_value[logearnStats.names.indexOf("beta[1]")];
    const beta1Ref = logearnStats.mean_value[logearnStats.names.indexOf("beta[2]")];
    const sigmaRef = logearnStats.mean_value[logearnStats.names.indexOf("sigma")];

    expectMeanClose("beta0", beta0Mean, beta0Ref);
    expectMeanClose("beta1", beta1Mean, beta1Ref);
    expectMeanClose("sigma", sigmaMean, sigmaRef);
    disposeHmcResult(result);
    maybeDisposeTree(initialParams, "initialParams");
    logMemory("test end: logearn height");
  });

  test("earn height", async () => {
    logMemory("test start: earn height");
    const earnStats = loadMeanStats("earnings-earn_height");
    const data = loadData("earnings");
    const bound = earnHeight.bind({
      height: np.array(data.height as number[]),
      earn: np.array(data.earn as number[]),
    });
    const initialParams = {
      beta0: np.array(0),
      beta1: np.array(0),
      sigma: np.array(0),
    };

    const result = await hmc(bound.logProb, {
      ...HMC_CONFIG,
      key: random.key(6),
      initialParams,
    });

    const beta0Mean = drawMean(result.draws.beta0);
    const beta1Mean = drawMean(result.draws.beta1);
    const sigmaMean = drawMean(result.draws.sigma);

    const beta0Ref = earnStats.mean_value[earnStats.names.indexOf("beta[1]")];
    const beta1Ref = earnStats.mean_value[earnStats.names.indexOf("beta[2]")];
    const sigmaRef = earnStats.mean_value[earnStats.names.indexOf("sigma")];

    expectMeanClose("beta0", beta0Mean, beta0Ref);
    expectMeanClose("beta1", beta1Mean, beta1Ref);
    expectMeanClose("sigma", sigmaMean, sigmaRef);
    disposeHmcResult(result);
    maybeDisposeTree(initialParams, "initialParams");
    logMemory("test end: earn height");
  });

  test("mesquite logvolume", async () => {
    logMemory("test start: mesquite logvolume");
    const mesquiteStats = loadMeanStats("mesquite-logmesquite_logvolume");
    const data = loadData("mesquite");
    const bound = mesquiteLogVolume.bind({
      K: 2,
      diam1: np.array(data.diam1 as number[]),
      diam2: np.array(data.diam2 as number[]),
      canopy_height: np.array(data.canopy_height as number[]),
      log_weight: np.log(np.array(data.weight as number[])),
    });
    const initialParams = {
      beta: np.zeros([2]),
      sigma: np.array(0),
    };

    const result = await hmc(bound.logProb, {
      ...HMC_CONFIG,
      key: random.key(7),
      initialParams,
    });

    const beta0Mean = drawMean(np.take(result.draws.beta.ref, 0, -1));
    const beta1Mean = drawMean(np.take(result.draws.beta.ref, 1, -1));
    const sigmaMean = drawMean(result.draws.sigma);

    const beta0Ref = mesquiteStats.mean_value[mesquiteStats.names.indexOf("beta[1]")];
    const beta1Ref = mesquiteStats.mean_value[mesquiteStats.names.indexOf("beta[2]")];
    const sigmaRef = mesquiteStats.mean_value[mesquiteStats.names.indexOf("sigma")];

    expectMeanClose("beta0", beta0Mean, beta0Ref);
    expectMeanClose("beta1", beta1Mean, beta1Ref);
    expectMeanClose("sigma", sigmaMean, sigmaRef);
    disposeHmcResult(result);
    maybeDisposeTree(initialParams, "initialParams");
    logMemory("test end: mesquite logvolume");
  });

  test("radon pooled", async () => {
    logMemory("test start: radon pooled");
    const radonStats = loadMeanStats("radon_all-radon_pooled");
    const data = loadData("radon_all");
    const bound = radonPooled.bind({
      floor_measure: np.array(data.floor_measure as number[]),
      log_radon: np.array(data.log_radon as number[]),
    });
    const initialParams = {
      alpha: np.array(0),
      beta: np.array(0),
      sigma_y: np.array(0),
    };

    const result = await hmc(bound.logProb, {
      ...HMC_CONFIG,
      key: random.key(8),
      initialParams,
    });

    const alphaMean = drawMean(result.draws.alpha);
    const betaMean = drawMean(result.draws.beta);
    const sigmaMean = drawMean(result.draws.sigma_y);

    const alphaRef = radonStats.mean_value[radonStats.names.indexOf("alpha")];
    const betaRef = radonStats.mean_value[radonStats.names.indexOf("beta")];
    const sigmaRef = radonStats.mean_value[radonStats.names.indexOf("sigma_y")];

    expectMeanClose("alpha", alphaMean, alphaRef);
    expectMeanClose("beta", betaMean, betaRef);
    expectMeanClose("sigma", sigmaMean, sigmaRef);
    disposeHmcResult(result);
    maybeDisposeTree(initialParams, "initialParams");
    logMemory("test end: radon pooled");
  });

  test("radon hierarchical", async () => {
    logMemory("test start: radon hierarchical");
    const radonHierStats = loadMeanStats("radon_mn-radon_hierarchical_intercept_noncentered");
    const data = loadData("radon_mn");
    const countyIdx = (data.county_idx as number[]).map((value) => value - 1);
    const bound = radonHierarchical.bind({
      J: data.J as number,
      county_idx: np.array(countyIdx, { dtype: np.int32 }),
      log_uppm: np.array(data.log_uppm as number[]),
      floor_measure: np.array(data.floor_measure as number[]),
      log_radon: np.array(data.log_radon as number[]),
    });
    const initialParams = {
      mu_alpha: np.array(0),
      sigma_alpha: np.array(0),
      sigma_y: np.array(0),
      alpha_raw: np.zeros([data.J as number]),
      beta_uppm: np.array(0),
      beta_floor: np.array(0),
    };

    const result = await hmc(bound.logProb, {
      ...HMC_CONFIG,
      key: random.key(9),
      initialParams,
    });

    const muAlphaMean = drawMean(result.draws.mu_alpha);
    const sigmaAlphaMean = drawMean(result.draws.sigma_alpha);
    const sigmaYMean = drawMean(result.draws.sigma_y);

    const muAlphaRef = radonHierStats.mean_value[radonHierStats.names.indexOf("mu_alpha")];
    const sigmaAlphaRef = radonHierStats.mean_value[radonHierStats.names.indexOf("sigma_alpha")];
    const sigmaYRef = radonHierStats.mean_value[radonHierStats.names.indexOf("sigma_y")];

    expectMeanClose("mu_alpha", muAlphaMean, muAlphaRef);
    expectMeanClose("sigma_alpha", sigmaAlphaMean, sigmaAlphaRef);
    expectMeanClose("sigma_y", sigmaYMean, sigmaYRef);
    disposeHmcResult(result);
    maybeDisposeTree(initialParams, "initialParams");
    logMemory("test end: radon hierarchical");
  });

  test("wells distance", async () => {
    logMemory("test start: wells distance");
    const wellsStats = loadMeanStats("wells_data-wells_dist");
    const data = loadData("wells_data");
    const bound = wellsDist.bind({
      dist: np.array(data.dist as number[]),
      switched: np.array(data.switched as number[]),
    });
    const initialParams = {
      beta0: np.array(0),
      beta1: np.array(0),
    };

    const result = await hmc(bound.logProb, {
      ...HMC_CONFIG,
      key: random.key(10),
      initialParams,
    });

    const beta0Mean = drawMean(result.draws.beta0);
    const beta1Mean = drawMean(result.draws.beta1);

    const beta0Ref = wellsStats.mean_value[wellsStats.names.indexOf("beta[1]")];
    const beta1Ref = wellsStats.mean_value[wellsStats.names.indexOf("beta[2]")];

    expectMeanClose("beta0", beta0Mean, beta0Ref);
    expectMeanClose("beta1", beta1Mean, beta1Ref);
    disposeHmcResult(result);
    maybeDisposeTree(initialParams, "initialParams");
    logMemory("test end: wells distance");
  });
});
