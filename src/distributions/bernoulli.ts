import { numpy as np, random, Array } from "@jax-js/jax";
import type { Distribution, DistributionSupport } from "./types";

/**
 * Bernoulli distribution.
 *
 * @param p - Probability of success (1). Must be in [0, 1].
 * @returns Distribution object with logProb and sample methods.
 */
export function bernoulli(p: number | Array): Distribution {
  const pArr = typeof p === "number" ? np.array(p) : p;

  const support: DistributionSupport = { type: "discrete" };

  return {
    logProb(x: Array): Array {
      // logProb = x * log(p) + (1 - x) * log(1 - p)
      const logP = np.log(pArr.ref);
      const log1MinusP = np.log(np.array(1).sub(pArr.ref));
      const logProbPerPoint = x.ref
        .mul(logP)
        .add(np.array(1).sub(x).mul(log1MinusP));
      return np.sum(logProbPerPoint);
    },

    sample(key: Array, shape: number[] = []): Array {
      // random.bernoulli returns boolean, convert to float for consistency
      const samples = random.bernoulli(key, pArr.ref, shape);
      return np.where(samples, np.array(1), np.array(0));
    },

    support,
  };
}
