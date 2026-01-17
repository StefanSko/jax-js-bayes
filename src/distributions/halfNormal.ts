import { Array as JaxArray, numpy as np } from "@jax-js/jax";
import type { Distribution } from "../types";
import { normal } from "./normal";

type ArrayLike = JaxArray | number | number[] | boolean | boolean[];

export function halfNormal(scale: ArrayLike): Distribution {
  const base = normal(0, scale);
  return {
    logProb(x: JaxArray) {
      const logProb = base.logProb(x.ref);
      const mask = np.greater(x, 0);
      return np.where(mask, logProb.add(Math.log(2)), -Infinity);
    },
    sample(key: JaxArray, shape: number[] = []) {
      return np.abs(base.sample(key, shape));
    },
  };
}
