import { numpy as np, type Array } from "@jax-js/jax";
import type { Constraint } from "./types";

export function positive(): Constraint {
  return {
    transform(unconstrained: Array): Array {
      return np.exp(unconstrained);
    },

    inverse(constrained: Array): Array {
      return np.log(constrained);
    },

    logDetJacobian(unconstrained: Array): Array {
      // d/dx exp(x) = exp(x), so log|det J| = sum(x) for element-wise transform
      // Returns scalar (sum over all dimensions)
      return unconstrained.sum();
    },
  };
}
