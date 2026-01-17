import { numpy as np, Array as JaxArray } from "@jax-js/jax";

export function meanValue(x: JaxArray): number {
  const xRef = x.ref;
  const numeric = xRef.dtype === np.bool ? xRef.astype(np.float32) : xRef;
  return np.mean(numeric).item();
}

export function stdValue(x: JaxArray): number {
  return np.std(x.ref).item();
}
