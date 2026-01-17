import { numpy as np, random, Array } from "@jax-js/jax";
import type { Distribution, DistributionSupport } from "./types";

/**
 * Uniform distribution over [low, high).
 *
 * @param low - Lower bound of the interval.
 * @param high - Upper bound of the interval.
 * @returns Distribution object with logProb and sample methods.
 */
export function uniform(low: number, high: number): Distribution {
  const range = high - low;
  const logDensity = -Math.log(range);

  const support: DistributionSupport = { type: "bounded", low, high };

  return {
    logProb(x: Array): Array {
      // logProb = -log(high - low) for x in [low, high), -inf otherwise
      // For simplicity, we assume x is always in bounds (HMC shouldn't leave valid region)
      const n = np.shape(x).reduce((a, b) => a * b, 1);
      return np.array(n * logDensity);
    },

    sample(key: Array, shape: number[] = []): Array {
      return random.uniform(key, shape, { minval: low, maxval: high });
    },

    support,
  };
}
