import { test } from "vitest";
import { numpy as np, random, tree } from "@jax-js/jax";
import { hmc } from "jax-js-mcmc";
import { HMC } from "jax-js-mcmc-2";
import { model, param, observed } from "../../src/index";
import { normal, halfNormal } from "../../src/distributions";
import { positive } from "../../src/constraints";

const enabled = process.env.HMC_MEM_REPRO === "1";
const maybeTest = enabled ? test : test.skip;

const MODE = process.env.HMC_MEM_MODE ?? "dsl"; // "dsl" | "raw"
const ITERATIONS = Number(process.env.HMC_MEM_ITERATIONS ?? 10);
const LOG_EVERY = Number(process.env.HMC_MEM_LOG_EVERY ?? 1);
const NUM_SAMPLES = Number(process.env.HMC_MEM_NUM_SAMPLES ?? 50);
const NUM_WARMUP = Number(process.env.HMC_MEM_NUM_WARMUP ?? 50);
const NUM_CHAINS = Number(process.env.HMC_MEM_NUM_CHAINS ?? 1);
const RAW_DIM = Number(process.env.HMC_MEM_RAW_DIM ?? 8);
const REUSE_DATA = process.env.HMC_MEM_REUSE_DATA !== "0";
const USE_GC = process.env.HMC_MEM_GC === "1";
const REUSE_SAMPLER = process.env.HMC_MEM_REUSE_SAMPLER === "1";
const STEP_SIZE = Number(process.env.HMC_MEM_STEP_SIZE ?? 0.1);
const NUM_INTEGRATION_STEPS = Number(process.env.HMC_MEM_NUM_INTEGRATION_STEPS ?? 10);

function formatMB(bytes: number): string {
  return `${(bytes / 1024 / 1024).toFixed(1)}MB`;
}

function logMemory(label: string): void {
  if (USE_GC && typeof global.gc === "function") {
    global.gc();
  }
  const { heapUsed, heapTotal, rss, external, arrayBuffers } = process.memoryUsage();
  const parts = [
    `heap=${formatMB(heapUsed)}/${formatMB(heapTotal)}`,
    `rss=${formatMB(rss)}`,
    `ext=${formatMB(external)}`,
  ];
  if (typeof arrayBuffers === "number") {
    parts.push(`ab=${formatMB(arrayBuffers)}`);
  }
  console.log(`[repro] ${label} ${parts.join(" ")}`);
}

function disposeTree(value: unknown): void {
  try {
    tree.dispose(value as never);
  } catch {
    // best-effort for repro; avoid failing the run
  }
}

const simpleModel = model({
  mu: param(normal(0, 1)),
  sigma: param(halfNormal(1), { constraint: positive() }),
  y: observed(({ mu, sigma }) => normal(mu, sigma)),
});

function buildDslBound() {
  const y = np.array([0.1, -0.2, 0.05, 0.3, -0.15]);
  const bound = simpleModel.bind({ y });
  return { bound, y };
}

function rawLogProb(params: { x: ReturnType<typeof np.array> }) {
  // Simple standard normal log-prob: -0.5 * sum(x^2)
  const x = params.x;
  return x.ref.mul(x).mul(-0.5).sum();
}

function rawLogDensity(x: ReturnType<typeof np.array>) {
  // Same as rawLogProb but for flat position arrays
  return x.ref.mul(x).mul(-0.5).sum();
}

async function runIteration(
  iter: number,
  boundData?: { bound: ReturnType<typeof simpleModel.bind>; y: ReturnType<typeof np.array> },
) {
  const key = random.key(iter);
  const initialParams = MODE === "raw"
    ? { x: np.zeros([RAW_DIM]) }
    : { mu: np.array(0), sigma: np.array(0) };

  const logProb = MODE === "raw"
    ? (rawLogProb as (p: Record<string, unknown>) => ReturnType<typeof np.array>)
    : boundData!.bound.logProb;

  const result = await hmc(logProb, {
    numSamples: NUM_SAMPLES,
    numWarmup: NUM_WARMUP,
    numChains: NUM_CHAINS,
    key,
    initialParams: initialParams as Record<string, unknown>,
  });

  disposeTree(result.draws);
  const massMatrix = result.stats.massMatrix;
  if (Array.isArray(massMatrix)) {
    for (const entry of massMatrix) {
      disposeTree(entry);
    }
  } else {
    disposeTree(massMatrix);
  }
  disposeTree(initialParams);
}

async function runIterationWithSampler(
  iter: number,
  sampler: ReturnType<ReturnType<typeof HMC>["build"]>,
) {
  let key = random.key(iter);
  let position = np.zeros([RAW_DIM]);
  const totalSteps = NUM_SAMPLES + Math.max(0, NUM_WARMUP);

  for (let i = 0; i < totalSteps; i++) {
    const [iterKey, nextKey] = random.split(key);
    key = nextKey;

    const state = sampler.init(position.ref);
    const [newState, info] = sampler.step(iterKey, state);

    try { info.momentum.dispose(); } catch {}
    try { info.acceptanceProb.dispose(); } catch {}
    try { info.isAccepted.dispose(); } catch {}
    try { info.isDivergent.dispose(); } catch {}
    try { info.energy.dispose(); } catch {}

    try { state.position.dispose(); } catch {}
    try { state.logdensity.dispose(); } catch {}
    try { state.logdensityGrad.dispose(); } catch {}

    try { newState.logdensity.dispose(); } catch {}
    try { newState.logdensityGrad.dispose(); } catch {}

    try { position.dispose(); } catch {}
    position = newState.position;
  }

  try { position.dispose(); } catch {}
  try { key.dispose(); } catch {}
}

maybeTest("hmc memory repro", async () => {
  if (REUSE_SAMPLER && MODE !== "raw") {
    throw new Error("HMC_MEM_REUSE_SAMPLER=1 only supported with HMC_MEM_MODE=raw");
  }

  logMemory(
    `start mode=${MODE} iters=${ITERATIONS} reuseSampler=${REUSE_SAMPLER}`,
  );

  let boundData: { bound: ReturnType<typeof simpleModel.bind>; y: ReturnType<typeof np.array> } | undefined;
  if (MODE === "dsl" && REUSE_DATA) {
    boundData = buildDslBound();
  }

  if (REUSE_SAMPLER) {
    const massMatrix = np.ones([RAW_DIM]);
    const sampler = HMC(rawLogDensity)
      .stepSize(STEP_SIZE)
      .numIntegrationSteps(NUM_INTEGRATION_STEPS)
      .inverseMassMatrix(massMatrix.ref)
      .jitStep(false)
      .build();

    for (let i = 0; i < ITERATIONS; i++) {
      await runIterationWithSampler(i, sampler);
      if ((i + 1) % LOG_EVERY === 0) {
        logMemory(`iter ${i + 1}`);
      }
    }

    try { massMatrix.dispose(); } catch {}
  } else {
    for (let i = 0; i < ITERATIONS; i++) {
      const iterationData = MODE === "dsl" && !REUSE_DATA ? buildDslBound() : boundData;
      await runIteration(i, iterationData);
      if (MODE === "dsl" && !REUSE_DATA && iterationData) {
        disposeTree(iterationData.y);
      }
      if ((i + 1) % LOG_EVERY === 0) {
        logMemory(`iter ${i + 1}`);
      }
    }
  }

  if (MODE === "dsl" && REUSE_DATA && boundData) {
    disposeTree(boundData.y);
  }

  logMemory("end");
}, 60_000 * Math.max(1, ITERATIONS));
