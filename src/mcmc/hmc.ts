import { numpy as np, random, grad, Array as JaxArray, type JsTree } from "@jax-js/jax";
import { HMC } from "jax-js-mcmc-2";
import {
  mapTree,
  treeClone,
  treeDispose,
  treeRef,
  treeFlatten,
  treeUnflatten,
  stackTrees,
  flattenToArray,
  unflattenFromArray,
  prod,
  treeOnesLike,
} from "./tree-utils";

export interface HMCOptions<Params extends JsTree<JaxArray>> {
  numSamples: number;
  initialParams: Params;
  key: JaxArray;

  numWarmup?: number;
  numLeapfrogSteps?: number;
  numChains?: number;

  initialStepSize?: number;
  targetAcceptRate?: number;

  adaptMassMatrix?: boolean;
}

export interface HMCStats<Params extends JsTree<JaxArray>> {
  acceptRate: number;
  acceptRatePerChain: number[];
  stepSize: number;
  stepSizePerChain: number[];
  massMatrix: Params | Params[];
}

export interface HMCResult<Params extends JsTree<JaxArray>> {
  draws: Params;
  stats: HMCStats<Params>;
}

function splitKeys(key: JaxArray, n: number): JaxArray[] {
  const keys: JaxArray[] = [];
  let current = key;
  for (let i = 0; i < n; i++) {
    const [k1, k2] = random.split(current);
    keys.push(k1);
    current = k2;
  }
  current.dispose();
  return keys;
}

function initMassMatrix<Params extends JsTree<JaxArray>>(params: Params): Params {
  return treeOnesLike(params);
}

function clampStepSize(stepSize: number): number {
  const MIN_STEP = 1e-10;
  const MAX_STEP = 1e10;
  return Math.max(MIN_STEP, Math.min(MAX_STEP, stepSize));
}

interface DualAverageState {
  logStepSize: number;
  logStepSizeAvg: number;
  hBar: number;
  mu: number;
  t: number;
  gamma: number;
  t0: number;
  kappa: number;
}

function initDualAverage(stepSize: number): DualAverageState {
  return {
    logStepSize: Math.log(stepSize),
    logStepSizeAvg: 0,
    hBar: 0,
    mu: Math.log(10 * stepSize),
    t: 0,
    gamma: 0.05,
    t0: 10,
    kappa: 0.75,
  };
}

function updateDualAverage(
  state: DualAverageState,
  acceptProb: number,
  targetAcceptRate: number,
): DualAverageState {
  const t = state.t + 1;
  const eta = 1 / (t + state.t0);
  const hBar = (1 - eta) * state.hBar + eta * (targetAcceptRate - acceptProb);
  const logStepSize = state.mu - (Math.sqrt(t) / state.gamma) * hBar;
  const tPow = Math.pow(t, -state.kappa);
  const logStepSizeAvg = tPow * logStepSize + (1 - tPow) * state.logStepSizeAvg;
  return { ...state, t, hBar, logStepSize, logStepSizeAvg };
}

interface MassMatrixState {
  count: number;
  mean: number[];
  m2: number[];
}

function initMassMatrixState<Params extends JsTree<JaxArray>>(params: Params): MassMatrixState {
  const [leaves] = treeFlatten(params);
  const totalSize = leaves.reduce((acc, leaf) => acc + prod(leaf.shape as number[]), 0);
  return {
    count: 0,
    mean: Array.from({ length: totalSize }, () => 0),
    m2: Array.from({ length: totalSize }, () => 0),
  };
}

function updateMassMatrixState<Params extends JsTree<JaxArray>>(
  state: MassMatrixState,
  position: Params,
): MassMatrixState {
  const flat = flattenToArray(treeRef(position));
  const values = flat.js() as number[];
  try { flat.dispose(); } catch {}

  const count = state.count + 1;
  const mean = [...state.mean];
  const m2 = [...state.m2];

  for (let i = 0; i < values.length; i++) {
    const delta = values[i] - mean[i];
    mean[i] += delta / count;
    const delta2 = values[i] - mean[i];
    m2[i] += delta * delta2;
  }

  return { count, mean, m2 };
}

function finalizeMassMatrix<Params extends JsTree<JaxArray>>(
  state: MassMatrixState,
  template: Params,
): Params {
  const variance = state.m2.map((m2, i) => {
    if (state.count < 2) return 1;
    const v = m2 / (state.count - 1);
    return Math.max(1e-3, v);
  });

  const flatMass = np.array(variance);
  const mass = unflattenFromArray(flatMass, template);
  // flatMass is consumed by unflattenFromArray (np.split)
  return mass;
}

async function runChain<Params extends JsTree<JaxArray>>(
  logProb: (p: Params) => JaxArray,
  options: Required<HMCOptions<Params>>,
  chainKey: JaxArray,
): Promise<{
  draws: Params;
  acceptRate: number;
  stepSize: number;
  massMatrix: Params;
}> {
  const {
    numSamples,
    numWarmup,
    numLeapfrogSteps,
    initialStepSize,
    targetAcceptRate,
    adaptMassMatrix,
    initialParams,
  } = options;

  let key = chainKey;
  let position = treeClone(initialParams);
  let massMatrix = initMassMatrix(initialParams);
  let massState = initMassMatrixState(initialParams);
  let stepSize = clampStepSize(initialStepSize);
  let dualState = initDualAverage(stepSize);

  const samples: Params[] = [];
  let acceptCount = 0;

  // Create the wrapped logProb once (reused across all iterations)
  const wrappedLogProb = (flatPos: JaxArray): JaxArray => {
    const params = unflattenFromArray(flatPos, treeRef(initialParams));
    const lp = logProb(params);
    return lp;
  };

  const totalIters = numWarmup + numSamples;

  // Current sampler (rebuilt when stepSize/massMatrix changes)
  let currentSampler: ReturnType<ReturnType<typeof HMC>["build"]> | null = null;
  let currentFlatMass: JaxArray | null = null;
  let lastStepSize = -1;
  let samplerNeedsRebuild = true;

  for (let iter = 0; iter < totalIters; iter++) {
    const [iterKey, nextKey] = random.split(key);
    key = nextKey;

    const flatPosition = flattenToArray(treeRef(position));

    // Rebuild sampler if stepSize changed or first iteration
    const inSamplingPhase = iter >= numWarmup;
    if (samplerNeedsRebuild || (!inSamplingPhase && stepSize !== lastStepSize)) {
      if (currentFlatMass) {
        try { currentFlatMass.dispose(); } catch {}
      }
      currentFlatMass = flattenToArray(treeRef(massMatrix));

      // Use JIT only during sampling phase (fixed parameters)
      // During warmup, parameters change so we use eager mode
      currentSampler = HMC(wrappedLogProb)
        .stepSize(stepSize)
        .numIntegrationSteps(numLeapfrogSteps)
        .inverseMassMatrix(currentFlatMass.ref)
        .jitStep(inSamplingPhase)
        .build();

      lastStepSize = stepSize;
      samplerNeedsRebuild = false;
    }

    const state = currentSampler!.init(flatPosition.ref);
    const [newState, info] = currentSampler!.step(iterKey, state);

    const acceptProbData = await info.acceptanceProb.data();
    const isAcceptedData = await info.isAccepted.data();
    const acceptProb = acceptProbData[0];
    const isAccepted = isAcceptedData[0] > 0.5;

    // Dispose info - note: some may already be consumed by data() call
    try { info.momentum.dispose(); } catch {}
    try { info.acceptanceProb.dispose(); } catch {}
    try { info.isAccepted.dispose(); } catch {}
    try { info.isDivergent.dispose(); } catch {}
    try { info.energy.dispose(); } catch {}

    try { state.position.dispose(); } catch {}
    try { state.logdensity.dispose(); } catch {}
    try { state.logdensityGrad.dispose(); } catch {}

    const newPosition = unflattenFromArray(newState.position, treeRef(initialParams));

    try { newState.logdensity.dispose(); } catch {}
    try { newState.logdensityGrad.dispose(); } catch {}

    try { flatPosition.dispose(); } catch {}

    treeDispose(position);
    position = newPosition;

    if (iter < numWarmup) {
      dualState = updateDualAverage(dualState, acceptProb, targetAcceptRate);
      stepSize = clampStepSize(Math.exp(dualState.logStepSize));

      if (adaptMassMatrix) {
        massState = updateMassMatrixState(massState, position);
      }

      if (iter === numWarmup - 1) {
        stepSize = clampStepSize(Math.exp(dualState.logStepSizeAvg));
        if (adaptMassMatrix) {
          treeDispose(massMatrix);
          massMatrix = finalizeMassMatrix(massState, initialParams);
          samplerNeedsRebuild = true; // Rebuild sampler with new mass matrix
        }
      }
    } else {
      if (isAccepted) acceptCount++;
      samples.push(treeClone(position));
    }
  }

  const acceptRate = numSamples > 0 ? acceptCount / numSamples : 0;
  const draws = stackTrees(samples, 0);
  // samples are consumed by stackTrees (np.stack)

  try { treeDispose(position); } catch {}
  if (currentFlatMass) {
    try { currentFlatMass.dispose(); } catch {}
  }

  return { draws, acceptRate, stepSize, massMatrix };
}

export async function hmc<Params extends JsTree<JaxArray>>(
  logProb: (p: Params) => JaxArray,
  options: HMCOptions<Params>,
): Promise<HMCResult<Params>> {
  if (options.numSamples <= 0) {
    throw new Error("numSamples must be > 0");
  }
  const opts: Required<HMCOptions<Params>> = {
    numSamples: options.numSamples,
    initialParams: options.initialParams,
    key: options.key,
    numWarmup: options.numWarmup ?? 1000,
    numLeapfrogSteps: options.numLeapfrogSteps ?? 25,
    numChains: options.numChains ?? 1,
    initialStepSize: options.initialStepSize ?? 0.1,
    targetAcceptRate: options.targetAcceptRate ?? 0.8,
    adaptMassMatrix: options.adaptMassMatrix ?? true,
  };

  const chainKeys = splitKeys(opts.key, opts.numChains);
  const chainResults = [] as {
    draws: Params;
    acceptRate: number;
    stepSize: number;
    massMatrix: Params;
  }[];

  for (let i = 0; i < opts.numChains; i++) {
    chainResults.push(await runChain(logProb, opts, chainKeys[i]));
  }

  const draws = stackTrees(
    chainResults.map((c) => c.draws),
    0,
  );
  // draws from chainResults are consumed by stackTrees

  const acceptRatePerChain = chainResults.map((c) => c.acceptRate);
  const stepSizePerChain = chainResults.map((c) => c.stepSize);
  const acceptRate =
    acceptRatePerChain.reduce((a, b) => a + b, 0) /
    (acceptRatePerChain.length || 1);
  const stepSize =
    stepSizePerChain.reduce((a, b) => a + b, 0) /
    (stepSizePerChain.length || 1);

  const stats: HMCStats<Params> = {
    acceptRate,
    acceptRatePerChain,
    stepSize,
    stepSizePerChain,
    massMatrix:
      opts.numChains === 1
        ? chainResults[0].massMatrix
        : chainResults.map((c) => c.massMatrix),
  };

  return { draws, stats };
}
