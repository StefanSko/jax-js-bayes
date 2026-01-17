import { numpy as np, random, type Array } from "@jax-js/jax";
import type { Distribution } from "./types";

export function uniform(low: number | Array = 0, high: number | Array = 1): Distribution {
  const lowArr = typeof low === "number" ? np.array(low) : low;
  const highArr = typeof high === "number" ? np.array(high) : high;

  return {
    logProb(x: Array): Array {
      // Uniform: f(x) = 1/(high-low) for low <= x <= high
      // log f(x) = -log(high-low)
      // We return -Infinity outside bounds, but for simplicity assume valid inputs
      const range = highArr.ref.sub(lowArr.ref);
      return np.log(range).mul(-1);
    },

    sample(key: Array, shape: number[] = []): Array {
      const range = highArr.ref.sub(lowArr.ref);
      return random.uniform(key, shape).mul(range).add(lowArr.ref);
    },
  };
}
