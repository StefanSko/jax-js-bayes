import { numpy as np, random, Array } from "@jax-js/jax";
import type { Distribution, DistributionSupport } from "./types";

/**
 * Exponential distribution.
 *
 * @param rate - Rate parameter (inverse of mean). Must be positive.
 * @returns Distribution object with logProb and sample methods.
 */
export function exponential(rate: number | Array): Distribution {
  const rateArr = typeof rate === "number" ? np.array(rate) : rate;

  const support: DistributionSupport = { type: "positive" };

  return {
    logProb(x: Array): Array {
      // logProb = log(rate) - rate * x
      const logProbPerPoint = np.log(rateArr.ref).sub(rateArr.ref.mul(x));
      return np.sum(logProbPerPoint);
    },

    sample(key: Array, shape?: number[]): Array {
      // Exponential(rate) = Exponential(1) / rate
      const sampleShape = shape ?? rateArr.shape;
      const z = random.exponential(key, sampleShape);
      return z.div(rateArr.ref);
    },

    support,
  };
}
