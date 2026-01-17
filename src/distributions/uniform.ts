import { Array as JaxArray, numpy as np, random } from "@jax-js/jax";
import type { Distribution } from "../types";
import { asArray, resolveSampleShape } from "./utils";

type ArrayLike = JaxArray | number | number[] | boolean | boolean[];

export function uniform(low: ArrayLike, high: ArrayLike): Distribution {
  return {
    logProb(x: JaxArray) {
      const lowArr = asArray(low);
      const highArr = asArray(high);
      const range = highArr.ref.sub(lowArr.ref);
      const logRange = np.log(range);
      const lowerOk = np.greaterEqual(x.ref, lowArr);
      const upperOk = np.lessEqual(x, highArr);
      const inside = np.where(lowerOk, upperOk, false);
      return np.where(inside, logRange.mul(-1), -Infinity);
    },
    sample(key: JaxArray, shape: number[] = []) {
      const lowArr = asArray(low);
      const highArr = asArray(high);
      const outputShape = resolveSampleShape(shape, lowArr, highArr);
      const base = random.uniform(key, outputShape);
      return base.mul(highArr.sub(lowArr.ref)).add(lowArr);
    },
  };
}
