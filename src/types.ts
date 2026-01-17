import type { Array } from "@jax-js/jax";

export type ShapeToken = number | string;
export type ShapeSpec = ShapeToken | ShapeToken[];

export type ModelContext = Record<string, Array>;

export interface Distribution {
  logProb(x: Array): Array;
  sample(key: Array, shape?: number[]): Array;
}

export type DistributionFactory = (ctx: ModelContext) => Distribution;

export interface Constraint {
  transform(unconstrained: Array): Array;
  inverse(constrained: Array): Array;
  logDetJacobian(unconstrained: Array): Array;
}
