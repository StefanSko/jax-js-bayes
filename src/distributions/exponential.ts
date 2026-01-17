import { numpy as np, random, type Array } from "@jax-js/jax";
import type { Distribution } from "./types";

export function exponential(rate: number | Array = 1): Distribution {
  const rateArr = typeof rate === "number" ? np.array(rate) : rate;

  return {
    logProb(x: Array): Array {
      // Exponential: f(x) = rate * exp(-rate * x) for x >= 0
      // log f(x) = log(rate) - rate * x
      return np.log(rateArr.ref).sub(rateArr.ref.mul(x));
    },

    sample(key: Array, shape: number[] = []): Array {
      // Inverse transform: x = -ln(U) / rate where U ~ Uniform(0,1)
      return random.exponential(key, shape).div(rateArr.ref);
    },
  };
}
