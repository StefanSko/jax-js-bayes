import { numpy as np, random, Array as JaxArray } from "@jax-js/jax";
import type { Distribution } from "../types";
import { sumToScalar, toArray } from "../utils";

const LOG_TWO = Math.log(2);
const LOG_TWO_PI = Math.log(2 * Math.PI);

type ArrayLike = JaxArray | number | number[];

export function halfNormal(scale: ArrayLike): Distribution {
  return {
    logProb(x: JaxArray): JaxArray {
      const xArr = toArray(x);
      const xRef = xArr.ref;
      const scaleArr = toArray(scale);
      const scaleRef = scaleArr.ref;
      const z = np.trueDivide(xArr, scaleArr);
      const logp = np.square(z)
        .mul(-0.5)
        .sub(np.log(scaleRef))
        .sub(0.5 * LOG_TWO_PI)
        .add(LOG_TWO);
      const masked = np.where(np.greaterEqual(xRef, 0), logp, -Infinity);
      return sumToScalar(masked);
    },
    sample(key: JaxArray, shape: number[] = []): JaxArray {
      const scaleArr = toArray(scale);
      return np.absolute(random.normal(key, shape)).mul(scaleArr);
    },
  };
}
