import { Array as JaxArray, numpy as np, random } from "@jax-js/jax";
import type { Distribution } from "../types";
import { asArray, resolveSampleShape } from "./utils";

type ArrayLike = JaxArray | number | number[] | boolean | boolean[];

export function exponential(rate: ArrayLike): Distribution {
  return {
    logProb(x: JaxArray) {
      const rateArr = asArray(rate);
      const logRate = np.log(rateArr.ref);
      const scaled = rateArr.mul(x);
      const base = logRate.sub(scaled);
      const mask = np.greaterEqual(x, 0);
      return np.where(mask, base, -Infinity);
    },
    sample(key: JaxArray, shape: number[] = []) {
      const rateArr = asArray(rate);
      const outputShape = resolveSampleShape(shape, rateArr);
      const draws = random.exponential(key, outputShape);
      return draws.div(rateArr);
    },
  };
}
