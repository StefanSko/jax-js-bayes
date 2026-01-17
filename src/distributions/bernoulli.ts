import { numpy as np, random, Array as JaxArray } from "@jax-js/jax";
import type { Distribution } from "../types";
import { sumToScalar, toArray } from "../utils";

type ArrayLike = JaxArray | number | number[];

export function bernoulli(p: ArrayLike): Distribution {
  return {
    logProb(x: JaxArray): JaxArray {
      const xArr = toArray(x);
      const mask = np.notEqual(xArr, 0);
      const pArr = toArray(p);
      const pRef = pArr.ref;
      const logp = np.where(mask, np.log(pArr), np.log(np.subtract(1, pRef)));
      return sumToScalar(logp);
    },
    sample(key: JaxArray, shape: number[] = []): JaxArray {
      return random.bernoulli(key, p, shape);
    },
  };
}
