import { numpy as np, random, Array } from "@jax-js/jax";
import type { Distribution, DistributionSupport } from "./types";

const LOG_2PI = Math.log(2 * Math.PI);
const LOG_2 = Math.log(2);

/**
 * Half-normal distribution.
 * This is |Normal(0, scale)| - the absolute value of a centered normal.
 *
 * @param scale - Scale parameter (standard deviation of the underlying normal).
 * @returns Distribution object with logProb and sample methods.
 */
export function halfNormal(scale: number | Array): Distribution {
  const scaleArr = typeof scale === "number" ? np.array(scale) : scale;

  const support: DistributionSupport = { type: "positive" };

  return {
    logProb(x: Array): Array {
      // logProb = log(2) + Normal(0, scale).logProb(x)
      // = log(2) - 0.5 * (x/scale)^2 - log(scale) - 0.5 * log(2π)
      const z = x.div(scaleArr.ref);
      const logProbPerPoint = z.ref
        .mul(z)
        .mul(-0.5)
        .sub(np.log(scaleArr.ref))
        .sub(0.5 * LOG_2PI)
        .add(LOG_2);
      return np.sum(logProbPerPoint);
    },

    sample(key: Array, shape: number[] = []): Array {
      const z = random.normal(key, shape);
      return np.abs(z.mul(scaleArr.ref));
    },

    support,
  };
}
