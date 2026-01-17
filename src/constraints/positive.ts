import { numpy as np } from "@jax-js/jax";
import type { Constraint } from "../types";

export function positive(): Constraint {
  return {
    transform: (x) => np.exp(x),
    inverse: (y) => np.log(y),
    logDetJacobian: (x) => x,
  };
}
