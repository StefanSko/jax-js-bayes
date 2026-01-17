import type { Array } from "@jax-js/jax";

export interface Distribution {
  logProb(x: Array): Array;
  sample(key: Array, shape?: number[]): Array;
}
