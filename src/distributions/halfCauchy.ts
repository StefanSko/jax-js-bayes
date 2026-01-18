import { numpy as np, random, Array as JaxArray } from "@jax-js/jax";
import type { Distribution } from "../types";
import { sumToScalar, toArray } from "../utils";

const LOG_TWO = Math.log(2);
const LOG_PI = Math.log(Math.PI);

type ArrayLike = JaxArray | number | number[];

export function halfCauchy(scale: ArrayLike): Distribution {
  return {
    logProb(x: JaxArray): JaxArray {
      const xArr = toArray(x);
      const xRef = xArr.ref;
      const scaleArr = toArray(scale);
      const scaleRef = scaleArr.ref;
      const z = np.trueDivide(xArr, scaleArr);
      const logp = np.log1p(np.square(z))
        .add(np.log(scaleRef))
        .add(LOG_PI)
        .mul(-1)
        .add(LOG_TWO);
      const masked = np.where(np.greaterEqual(xRef, 0), logp, -Infinity);
      return sumToScalar(masked);
    },
    sample(key: JaxArray, shape: number[] = []): JaxArray {
      const scaleArr = toArray(scale);
      return np.absolute(random.cauchy(key, shape)).mul(scaleArr);
    },
  };
}
