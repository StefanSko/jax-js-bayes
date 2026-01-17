import { numpy as np, type Array } from "@jax-js/jax";

/**
 * Index into a 1D array using integer indices.
 * NOTE: This uses np.take which does not support gradients in jax-js yet.
 * For models that need gradient through indexing, use explicit broadcasting.
 */
export function take(arr: Array, indices: Array, axis: number = 0): Array {
  return np.take(arr, indices, axis);
}
