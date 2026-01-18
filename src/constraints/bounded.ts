import { numpy as np, Array } from "@jax-js/jax";
import type { Constraint } from "./types";

/**
 * Bounded constraint using sigmoid transform.
 * Maps unconstrained real line to bounded interval (low, high).
 *
 * transform: x → low + (high - low) * sigmoid(x)
 * inverse: y → logit((y - low) / (high - low))
 * logDetJacobian: x → log(high - low) + log(sigmoid(x)) + log(1 - sigmoid(x))
 *
 * @param low - Lower bound of the interval.
 * @param high - Upper bound of the interval.
 * @returns Constraint object.
 */
export function bounded(low: number, high: number): Constraint {
  const range = high - low;
  const logRange = Math.log(range);

  return {
    transform(unconstrained: Array): Array {
      // sigmoid(x) = 1 / (1 + exp(-x))
      const sig = np.array(1).div(np.exp(unconstrained.mul(-1)).add(1));
      return sig.mul(range).add(low);
    },

    inverse(constrained: Array): Array {
      // logit((y - low) / range) = log(p / (1 - p))
      const p = constrained.sub(low).div(range);
      // logit(p) = log(p) - log(1 - p)
      return np.log(p.ref).sub(np.log(np.array(1).sub(p)));
    },

    logDetJacobian(unconstrained: Array): Array {
      // d/dx [range * sigmoid(x)] = range * sigmoid(x) * (1 - sigmoid(x))
      // log|det(J)| = log(range) + log(sigmoid(x)) + log(1 - sigmoid(x))
      const sig = np.array(1).div(np.exp(unconstrained.mul(-1)).add(1));
      const logSig = np.log(sig.ref);
      const log1MinusSig = np.log(np.array(1).sub(sig));
      const logDetJPerPoint = logSig.add(log1MinusSig).add(logRange);
      return np.sum(logDetJPerPoint);
    },
  };
}
