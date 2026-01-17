import { nn, numpy as np, Array as JaxArray } from "@jax-js/jax";
import type { Constraint } from "../types";

function logit(x: JaxArray): JaxArray {
  const xRef = x.ref;
  return np.log(np.trueDivide(x, np.subtract(1, xRef)));
}

export function bounded(low: number, high: number): Constraint {
  const range = high - low;
  return {
    transform(x: JaxArray): JaxArray {
      return nn.sigmoid(x).mul(range).add(low);
    },
    inverse(y: JaxArray): JaxArray {
      return logit(np.trueDivide(np.subtract(y, low), range));
    },
    logDetJacobian(x: JaxArray): JaxArray {
      const s = nn.sigmoid(x);
      const sRef = s.ref;
      return np
        .log(s)
        .add(np.log(np.subtract(1, sRef)))
        .add(Math.log(range));
    },
  };
}
