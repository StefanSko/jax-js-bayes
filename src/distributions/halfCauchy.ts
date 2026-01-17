import { numpy as np, random, type Array } from "@jax-js/jax";
import type { Distribution } from "./types";

const LOG_2_OVER_PI = Math.log(2 / Math.PI);

export function halfCauchy(scale: number | Array = 1): Distribution {
  const scaleArr = typeof scale === "number" ? np.array(scale) : scale;

  return {
    logProb(x: Array): Array {
      // Half-Cauchy: f(x) = 2 / (pi * scale * (1 + (x/scale)^2)) for x > 0
      // log f(x) = log(2/pi) - log(scale) - log(1 + (x/scale)^2)
      const z = x.div(scaleArr.ref);
      const logOnePlusZ2 = np.log(z.ref.mul(z).add(1));
      return np.array(LOG_2_OVER_PI).sub(np.log(scaleArr.ref)).sub(logOnePlusZ2);
    },

    sample(key: Array, shape: number[] = []): Array {
      // Sample from standard Cauchy and take abs, then scale
      const u = random.cauchy(key, shape);
      return np.abs(u).mul(scaleArr.ref);
    },
  };
}
