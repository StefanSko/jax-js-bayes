import { Array as JaxArray, numpy as np } from "@jax-js/jax";
import type { Distribution } from "../types";
import { asArray, resolveSampleShape } from "./utils";
import { random } from "@jax-js/jax";

type ArrayLike = JaxArray | number | number[] | boolean | boolean[];

export function halfCauchy(scale: ArrayLike): Distribution {
  return {
    logProb(x: JaxArray) {
      const scaleArr = asArray(scale);
      const z = x.div(scaleArr.ref);
      const logDenom = np.log1p(z.pow(2));
      const logScale = np.log(scaleArr);
      const base = logDenom.add(logScale).add(Math.log(Math.PI)).mul(-1);
      const mask = np.greater(x, 0);
      return np.where(mask, base.add(Math.log(2)), -Infinity);
    },
    sample(key: JaxArray, shape: number[] = []) {
      const scaleArr = asArray(scale);
      const outputShape = resolveSampleShape(shape, scaleArr);
      const draws = random.cauchy(key, outputShape).mul(scaleArr);
      return np.abs(draws);
    },
  };
}
