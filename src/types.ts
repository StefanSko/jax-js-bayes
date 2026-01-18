import type { Array, JsTree } from "@jax-js/jax";
import type { Distribution } from "./distributions/types";
import type { Constraint } from "./constraints/types";

/**
 * Parameter specification in a model.
 */
export interface ParamSpec {
  type: "param";
  prior: Distribution;
  constraint?: Constraint;
  shape?: string | number[];
}

/**
 * Data declaration in a model.
 */
export interface DataSpec {
  type: "data";
  shape?: string | number[];
}

/**
 * Observed (likelihood) specification in a model.
 */
export interface ObservedSpec {
  type: "observed";
  likelihood: (params: Record<string, Array>) => Distribution;
}

/**
 * Derived quantity specification (computed from parameters).
 */
export type DerivedSpec = (params: Record<string, Array>) => Array;

/**
 * Model specification entry - can be param, data, observed, or derived.
 */
export type ModelEntry = ParamSpec | DataSpec | ObservedSpec | DerivedSpec;

/**
 * Model specification object.
 */
export type ModelSpec = Record<string, ModelEntry>;

/**
 * Bound model with data attached.
 * S is "complete" when all observed data is provided,
 * "predictive" when only covariates are provided.
 */
export interface BoundModel<S extends "complete" | "predictive"> {
  /**
   * Log probability function for HMC sampling.
   * Only available on complete models.
   */
  logProb: S extends "complete" ? (params: JsTree<Array>) => Array : never;

  /**
   * The bound data.
   */
  data: Record<string, Array>;

  /**
   * Model state type marker.
   */
  readonly state: S;
}

/**
 * Unbound model definition.
 */
export interface Model<Spec extends ModelSpec> {
  /**
   * Bind data to the model.
   * Returns a complete model if all observed variables are provided,
   * otherwise returns a predictive model.
   */
  bind<D extends Record<string, Array | number[]>>(
    data: D,
  ): BoundModel<"complete"> | BoundModel<"predictive">;

  /**
   * Sample parameters from the prior.
   * @param options - Options including PRNG key and optional shape dimensions.
   */
  samplePrior(options: { key: Array; dims?: Record<string, number> }): JsTree<Array>;

  /**
   * Simulate data from the model given parameters.
   * @param params - Parameter values.
   * @param options - Options including PRNG key.
   */
  simulate(
    params: JsTree<Array>,
    options: { key: Array },
  ): Record<string, Array>;

  /**
   * The original model specification.
   */
  readonly spec: Spec;
}
