import { numpy as np, random, Array as JaxArray } from "@jax-js/jax";
import type { Distribution } from "../types";
import { sumToScalar, toArray } from "../utils";

type ArrayLike = JaxArray | number | number[];

export function exponential(rate: ArrayLike): Distribution {
  return {
    logProb(x: JaxArray): JaxArray {
      const xArr = toArray(x);
      const xRef = xArr.ref;
      const rateArr = toArray(rate);
      const rateRef = rateArr.ref;
      const logp = np.log(rateRef).sub(np.multiply(rateArr, xArr));
      const masked = np.where(np.greaterEqual(xRef, 0), logp, -Infinity);
      return sumToScalar(masked);
    },
    sample(key: JaxArray, shape: number[] = []): JaxArray {
      return random.exponential(key, shape).div(rate);
    },
  };
}
