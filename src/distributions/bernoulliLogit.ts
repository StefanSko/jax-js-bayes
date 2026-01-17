import { numpy as np, random, nn, type Array } from "@jax-js/jax";
import type { Distribution } from "./types";

export function bernoulliLogit(logits: number | Array = 0): Distribution {
  const logitsArr = typeof logits === "number" ? np.array(logits) : logits;

  return {
    logProb(x: Array): Array {
      // Bernoulli with logit parameterization:
      // log P(x|logits) = x*logits - log(1 + exp(logits))
      //                 = x*logits - softplus(logits)
      // softplus(x) = log(1 + exp(x))
      const softplus = nn.softplus(logitsArr.ref);
      return x.mul(logitsArr.ref).sub(softplus);
    },

    sample(key: Array, shape: number[] = []): Array {
      const p = nn.sigmoid(logitsArr.ref);
      // bernoulli returns boolean, cast to float32 for numeric operations
      const boolSamples = random.bernoulli(key, p.ref, shape);
      return boolSamples.astype("float32");
    },
  };
}
