import { numpy as np, random, Array as JaxArray } from "@jax-js/jax";
import type { Distribution } from "../types";
import { sumToScalar, toArray } from "../utils";

const LOG_TWO_PI = Math.log(2 * Math.PI);

type ArrayLike = JaxArray | number | number[];

export function normal(loc: ArrayLike, scale: ArrayLike): Distribution {
  return {
    logProb(x: JaxArray): JaxArray {
      const xArr = toArray(x);
      const locArr = toArray(loc);
      const scaleArr = toArray(scale);
      const scaleRef = scaleArr.ref;
      const z = np.trueDivide(np.subtract(xArr, locArr), scaleArr);
      const logp = np.square(z)
        .mul(-0.5)
        .sub(np.log(scaleRef))
        .sub(0.5 * LOG_TWO_PI);
      return sumToScalar(logp);
    },
    sample(key: JaxArray, shape: number[] = []): JaxArray {
      const locArr = toArray(loc);
      const scaleArr = toArray(scale);
      return random.normal(key, shape).mul(scaleArr).add(locArr);
    },
  };
}
