import { Array as JaxArray, nn, numpy as np, random } from "@jax-js/jax";
import type { Distribution } from "../types";
import { asArray, resolveSampleShape } from "./utils";

type ArrayLike = JaxArray | number | number[] | boolean | boolean[];

export function bernoulliLogit(logit: ArrayLike): Distribution {
  return {
    logProb(x: JaxArray) {
      const logitArr = asArray(logit);
      const xFloat = x.astype(np.float32);
      const term = xFloat.mul(logitArr.ref);
      const norm = nn.softplus(logitArr);
      return term.sub(norm);
    },
    sample(key: JaxArray, shape: number[] = []) {
      const logitArr = asArray(logit);
      const outputShape = resolveSampleShape(shape, logitArr);
      const p = nn.sigmoid(logitArr);
      const draws = random.bernoulli(key, p, outputShape);
      return np.where(draws, 1, 0);
    },
  };
}
