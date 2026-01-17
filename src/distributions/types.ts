import type { Array } from "@jax-js/jax";

/**
 * Support types for distributions.
 * Used to determine what constraints are appropriate for parameters.
 */
export type DistributionSupport =
  | { type: "real" }
  | { type: "positive" }
  | { type: "bounded"; low: number; high: number }
  | { type: "discrete" };

/**
 * Distribution interface.
 * All distributions implement logProb and sample methods.
 */
export interface Distribution {
  /**
   * Log probability density/mass function.
   * @param x - Value(s) to evaluate. Can be scalar or batched.
   * @returns Scalar Array with summed log probability.
   */
  logProb(x: Array): Array;

  /**
   * Sample from the distribution.
   * @param key - PRNG key from jax-js random module.
   * @param shape - Optional batch shape for samples.
   * @returns Array of samples with given shape.
   */
  sample(key: Array, shape?: number[]): Array;

  /**
   * The support of this distribution.
   */
  readonly support: DistributionSupport;
}
