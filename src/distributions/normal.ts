import { numpy as np, random, Array } from "@jax-js/jax";
import type { Distribution, DistributionSupport } from "./types";

const LOG_2PI = Math.log(2 * Math.PI);

/**
 * Normal (Gaussian) distribution.
 *
 * @param loc - Mean of the distribution. Can be a number or Array.
 * @param scale - Standard deviation of the distribution. Can be a number or Array.
 * @returns Distribution object with logProb and sample methods.
 */
export function normal(
  loc: number | Array,
  scale: number | Array,
): Distribution {
  const locArr = typeof loc === "number" ? np.array(loc) : loc;
  const scaleArr = typeof scale === "number" ? np.array(scale) : scale;

  const support: DistributionSupport = { type: "real" };

  return {
    logProb(x: Array): Array {
      // logProb = -0.5 * ((x - loc) / scale)^2 - log(scale) - 0.5 * log(2π)
      const z = x.sub(locArr.ref).div(scaleArr.ref);
      const logProbPerPoint = z.ref
        .mul(z)
        .mul(-0.5)
        .sub(np.log(scaleArr.ref))
        .sub(0.5 * LOG_2PI);
      return np.sum(logProbPerPoint);
    },

    sample(key: Array, shape?: number[]): Array {
      const sampleShape = shape ?? np.broadcastShapes(locArr.shape, scaleArr.shape);
      const z = random.normal(key, sampleShape);
      return z.mul(scaleArr.ref).add(locArr.ref);
    },

    support,
  };
}
