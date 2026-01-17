import type { Array } from "@jax-js/jax";

export interface Constraint {
  transform(unconstrained: Array): Array;
  inverse(constrained: Array): Array;
  logDetJacobian(unconstrained: Array): Array;
}
