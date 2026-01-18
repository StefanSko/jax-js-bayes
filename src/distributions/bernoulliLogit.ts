import { nn, numpy as np, random, Array as JaxArray } from "@jax-js/jax";
import type { Distribution } from "../types";
import { sumToScalar, toArray } from "../utils";

type ArrayLike = JaxArray | number | number[];

export function bernoulliLogit(logits: ArrayLike): Distribution {
  return {
    logProb(x: JaxArray): JaxArray {
      const xArr = toArray(x);
      const mask = np.notEqual(xArr, 0);
      const logitsArr = toArray(logits);
      const logitsRef = logitsArr.ref;
      const logp = np.where(
        mask,
        nn.logSigmoid(logitsArr),
        nn.logSigmoid(np.multiply(logitsRef, -1)),
      );
      return sumToScalar(logp);
    },
    sample(key: JaxArray, shape: number[] = []): JaxArray {
      const logitsArr = toArray(logits);
      const probs = nn.sigmoid(logitsArr);
      return random.bernoulli(key, probs, shape);
    },
  };
}
