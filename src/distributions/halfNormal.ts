import { numpy as np, random, type Array } from "@jax-js/jax";
import type { Distribution } from "./types";

const LOG_2 = Math.log(2);
const LOG_SQRT_2PI = 0.5 * Math.log(2 * Math.PI);

export function halfNormal(scale: number | Array = 1): Distribution {
  const scaleArr = typeof scale === "number" ? np.array(scale) : scale;

  return {
    logProb(x: Array): Array {
      // HalfNormal: f(x) = 2/sqrt(2π)/scale * exp(-x²/(2*scale²)) for x >= 0
      // log f(x) = log(2) - log(sqrt(2π)) - log(scale) - x²/(2*scale²)
      const z = x.div(scaleArr.ref);
      return np
        .array(LOG_2 - LOG_SQRT_2PI)
        .sub(np.log(scaleArr.ref))
        .sub(z.ref.mul(z).mul(0.5));
    },

    sample(key: Array, shape: number[] = []): Array {
      return np.abs(random.normal(key, shape).mul(scaleArr.ref));
    },
  };
}
