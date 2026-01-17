import { numpy as np, Array } from "@jax-js/jax";
import type { Constraint } from "./types";

/**
 * Positive constraint using exp/log transform.
 * Maps unconstrained real line to positive reals (0, ∞).
 *
 * transform: x → exp(x)
 * inverse: y → log(y)
 * logDetJacobian: x → x (since d/dx exp(x) = exp(x), log|exp(x)| = x)
 *
 * @returns Constraint object.
 */
export function positive(): Constraint {
  return {
    transform(unconstrained: Array): Array {
      return np.exp(unconstrained);
    },

    inverse(constrained: Array): Array {
      return np.log(constrained);
    },

    logDetJacobian(unconstrained: Array): Array {
      // For exp transform: d/dx exp(x) = exp(x)
      // log|det(J)| = log(exp(x)) = x
      // Sum over all elements for batched inputs
      return np.sum(unconstrained);
    },
  };
}
