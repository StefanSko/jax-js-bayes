import { numpy as np, random, Array } from "@jax-js/jax";
import type { Distribution, DistributionSupport } from "./types";

/**
 * Bernoulli distribution parameterized by log-odds (logit).
 * p = sigmoid(logit) = 1 / (1 + exp(-logit))
 *
 * This is more numerically stable for logistic regression models.
 *
 * @param logit - Log-odds parameter. Can be any real number.
 * @returns Distribution object with logProb and sample methods.
 */
export function bernoulliLogit(logit: number | Array): Distribution {
  const logitArr = typeof logit === "number" ? np.array(logit) : logit;

  const support: DistributionSupport = { type: "discrete" };

  return {
    logProb(x: Array): Array {
      // logProb(x=1) = -log(1 + exp(-logit)) = -softplus(-logit)
      // logProb(x=0) = -log(1 + exp(logit)) = -softplus(logit)
      // Combined: x * logit - softplus(logit)
      // Where softplus(a) = log(1 + exp(a))

      // Numerically stable softplus: max(0, a) + log(1 + exp(-|a|))
      const absLogit = np.abs(logitArr.ref);
      const softplus = np.maximum(logitArr.ref, np.array(0)).add(
        np.log(np.exp(absLogit.mul(-1)).add(1)),
      );

      const logProbPerPoint = x.mul(logitArr.ref).sub(softplus);
      return np.sum(logProbPerPoint);
    },

    sample(key: Array, shape: number[] = []): Array {
      // p = sigmoid(logit) = 1 / (1 + exp(-logit))
      const p = np.array(1).div(np.exp(logitArr.ref.mul(-1)).add(1));
      const samples = random.bernoulli(key, p, shape);
      return np.where(samples, np.array(1), np.array(0));
    },

    support,
  };
}
