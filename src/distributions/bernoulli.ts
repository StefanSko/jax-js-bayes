import { Array as JaxArray, numpy as np, random } from "@jax-js/jax";
import type { Distribution } from "../types";
import { asArray, resolveSampleShape } from "./utils";

type ArrayLike = JaxArray | number | number[] | boolean | boolean[];

export function bernoulli(p: ArrayLike): Distribution {
  return {
    logProb(x: JaxArray) {
      const pArr = asArray(p);
      const logP = np.log(pArr.ref);
      const log1mP = np.log1p(pArr.mul(-1));
      const xFloat = x.astype(np.float32);
      const oneMinus = np.subtract(1, xFloat.ref);
      return xFloat.mul(logP).add(oneMinus.mul(log1mP));
    },
    sample(key: JaxArray, shape: number[] = []) {
      const pArr = asArray(p);
      const outputShape = resolveSampleShape(shape, pArr);
      const draws = random.bernoulli(key, pArr, outputShape);
      return np.where(draws, 1, 0);
    },
  };
}
