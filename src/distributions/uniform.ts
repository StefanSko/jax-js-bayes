import { numpy as np, random, Array as JaxArray } from "@jax-js/jax";
import type { Distribution } from "../types";
import { sumToScalar, toArray } from "../utils";

type ArrayLike = JaxArray | number | number[];

export function uniform(low: ArrayLike, high: ArrayLike): Distribution {
  return {
    logProb(x: JaxArray): JaxArray {
      const xArr = toArray(x);
      const lowArr = toArray(low);
      const highArr = toArray(high);
      const lowRef = lowArr.ref;
      const highRef = highArr.ref;
      const range = np.subtract(highArr, lowArr);
      const logp = np.log(range).mul(-1);
      const withinLow = np.greaterEqual(xArr.ref, lowRef);
      const withinHigh = np.lessEqual(xArr.ref, highRef);
      const masked = np.where(withinLow, np.where(withinHigh, logp, -Infinity), -Infinity);
      return sumToScalar(masked);
    },
    sample(key: JaxArray, shape: number[] = []): JaxArray {
      return random.uniform(key, shape).mul(np.subtract(high, low)).add(low);
    },
  };
}
