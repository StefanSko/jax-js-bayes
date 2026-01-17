import { numpy as np, nn, type Array } from "@jax-js/jax";
import type { Constraint } from "./types";

export function bounded(low: number = 0, high: number = 1): Constraint {
  const range = high - low;
  const logRange = Math.log(range);

  return {
    transform(unconstrained: Array): Array {
      // Sigmoid transform scaled to [low, high]
      return nn.sigmoid(unconstrained).mul(range).add(low);
    },

    inverse(constrained: Array): Array {
      // logit((constrained - low) / range)
      const normalized = constrained.sub(low).div(range);
      // logit(p) = log(p / (1-p)) = log(p) - log(1-p)
      return np.log(normalized.ref).sub(np.log(np.array(1).sub(normalized)));
    },

    logDetJacobian(unconstrained: Array): Array {
      // d/dx [range * sigmoid(x)] = range * sigmoid(x) * (1 - sigmoid(x))
      // log|det J| = sum(log(range) + log(sigmoid(x)) + log(1 - sigmoid(x)))
      const s = nn.sigmoid(unconstrained.ref);
      const logS = np.log(s.ref);
      const log1MinusS = np.log(np.array(1).sub(s));
      return logS.add(log1MinusS).add(logRange).sum();
    },
  };
}
