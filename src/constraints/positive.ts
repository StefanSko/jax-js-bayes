import { numpy as np, Array as JaxArray } from "@jax-js/jax";
import type { Constraint } from "../types";

export function positive(): Constraint {
  return {
    transform(x: JaxArray): JaxArray {
      return np.exp(x);
    },
    inverse(y: JaxArray): JaxArray {
      return np.log(y);
    },
    logDetJacobian(x: JaxArray): JaxArray {
      return x;
    },
  };
}
