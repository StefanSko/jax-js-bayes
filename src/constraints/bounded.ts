import { nn, numpy as np, scipySpecial } from "@jax-js/jax";
import type { Constraint } from "../types";

export function bounded(low: number, high: number): Constraint {
  const range = high - low;
  if (!(range > 0)) {
    throw new Error(`Invalid bounds: [${low}, ${high}]`);
  }
  return {
    transform: (x) => nn.sigmoid(x).mul(range).add(low),
    inverse: (y) => scipySpecial.logit(y.sub(low).div(range)),
    logDetJacobian: (x) => {
      const s = nn.sigmoid(x);
      const logS = np.log(s.ref);
      const logOneMinus = np.log(np.subtract(1, s));
      return logS.add(logOneMinus).add(Math.log(range));
    },
  };
}
