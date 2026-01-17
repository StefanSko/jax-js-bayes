import { numpy as np, random, Array } from "@jax-js/jax";
import type { Distribution, DistributionSupport } from "./types";

const LOG_2 = Math.log(2);
const LOG_PI = Math.log(Math.PI);

/**
 * Half-Cauchy distribution.
 * This is |Cauchy(0, scale)| - the absolute value of a centered Cauchy.
 * Has heavier tails than half-normal, commonly used as a prior for scale parameters.
 *
 * @param scale - Scale parameter of the underlying Cauchy distribution.
 * @returns Distribution object with logProb and sample methods.
 */
export function halfCauchy(scale: number | Array): Distribution {
  const scaleArr = typeof scale === "number" ? np.array(scale) : scale;

  const support: DistributionSupport = { type: "positive" };

  return {
    logProb(x: Array): Array {
      // Cauchy logProb = -log(π) - log(scale) - log(1 + (x/scale)^2)
      // Half-Cauchy = log(2) + Cauchy.logProb(x)
      const z = x.div(scaleArr.ref);
      const logProbPerPoint = np
        .log(z.ref.mul(z).add(1))
        .mul(-1)
        .sub(LOG_PI)
        .sub(np.log(scaleArr.ref))
        .add(LOG_2);
      return np.sum(logProbPerPoint);
    },

    sample(key: Array, shape: number[] = []): Array {
      // Use inverse CDF sampling for Cauchy
      // Cauchy: x = scale * tan(π * (u - 0.5))
      // Half-Cauchy: |x|
      const z = random.cauchy(key, shape);
      return np.abs(z.mul(scaleArr.ref));
    },

    support,
  };
}
