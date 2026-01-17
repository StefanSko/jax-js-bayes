import type { Array } from "@jax-js/jax";

/**
 * Constraint interface.
 * Constraints transform unconstrained parameters to constrained space
 * and provide Jacobian adjustments for HMC.
 */
export interface Constraint {
  /**
   * Transform from unconstrained to constrained space.
   * @param unconstrained - Value(s) in unconstrained space (real line).
   * @returns Value(s) in constrained space.
   */
  transform(unconstrained: Array): Array;

  /**
   * Transform from constrained to unconstrained space.
   * @param constrained - Value(s) in constrained space.
   * @returns Value(s) in unconstrained space (real line).
   */
  inverse(constrained: Array): Array;

  /**
   * Log absolute determinant of the Jacobian.
   * This is added to logProb for proper density transformation.
   * @param unconstrained - Value(s) in unconstrained space.
   * @returns Scalar Array with summed log|det(J)|.
   */
  logDetJacobian(unconstrained: Array): Array;
}
