import { Array as JaxArray, numpy as np, random } from "@jax-js/jax";
import type { Distribution } from "../types";
import { asArray, resolveSampleShape } from "./utils";

type ArrayLike = JaxArray | number | number[] | boolean | boolean[];

const LOG_SQRT_2PI = 0.5 * Math.log(2 * Math.PI);

export function normal(loc: ArrayLike, scale: ArrayLike): Distribution {
  return {
    logProb(x: JaxArray) {
      const locArr = asArray(loc);
      const scaleArr = asArray(scale);
      const z = x.sub(locArr).div(scaleArr.ref);
      const logScale = np.log(scaleArr);
      return np.square(z).mul(-0.5).sub(logScale).sub(LOG_SQRT_2PI);
    },
    sample(key: JaxArray, shape: number[] = []) {
      const locArr = asArray(loc);
      const scaleArr = asArray(scale);
      const outputShape = resolveSampleShape(shape, locArr, scaleArr);
      const z = random.normal(key, outputShape);
      return z.mul(scaleArr).add(locArr);
    },
  };
}
