import { numpy as np, random, type Array } from "@jax-js/jax";
import type { Distribution } from "./types";

const LOG_SQRT_2PI = 0.5 * Math.log(2 * Math.PI);

export function normal(
  loc: number | Array = 0,
  scale: number | Array = 1,
): Distribution {
  const locArr = typeof loc === "number" ? np.array(loc) : loc;
  const scaleArr = typeof scale === "number" ? np.array(scale) : scale;

  return {
    logProb(x: Array): Array {
      const z = x.sub(locArr.ref).div(scaleArr.ref);
      return z.ref
        .mul(z)
        .mul(-0.5)
        .sub(np.log(scaleArr.ref))
        .sub(LOG_SQRT_2PI);
    },

    sample(key: Array, shape: number[] = []): Array {
      return random.normal(key, shape).mul(scaleArr.ref).add(locArr.ref);
    },
  };
}
