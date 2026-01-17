import type { Array, JsTree } from "@jax-js/jax";
import type { Distribution } from "./distributions/types";
import type { Constraint } from "./constraints/types";

export const PARAM = Symbol("param");
export const DATA = Symbol("data");
export const OBSERVED = Symbol("observed");

export interface ParamSpec {
  kind: typeof PARAM;
  prior: Distribution;
  constraint?: Constraint;
  shape?: number[] | string;
}

export interface DataSpec {
  kind: typeof DATA;
  shape?: number[] | string;
}

export interface ObservedSpec {
  kind: typeof OBSERVED;
  distribution: (context: Record<string, Array>) => Distribution;
}

export type DerivedSpec = (context: Record<string, Array>) => Array;

export type ModelFieldSpec = ParamSpec | DataSpec | ObservedSpec | DerivedSpec;

export type ModelSpec = Record<string, ModelFieldSpec>;

export interface Model<Spec extends ModelSpec> {
  spec: Spec;
  bind<D extends Record<string, Array | number[]>>(
    data: D,
  ): BoundModel<HasAllObserved<Spec, D> extends true ? "complete" : "predictive">;
}

export interface BoundModel<State extends "complete" | "predictive"> {
  state: State;
  logProb: (params: JsTree<Array>) => Array;
  initialParams: () => JsTree<Array>;
  paramNames: string[];
}

type HasAllObserved<Spec extends ModelSpec, D extends Record<string, any>> = {
  [K in keyof Spec]: Spec[K] extends ObservedSpec
    ? K extends keyof D
      ? true
      : false
    : true;
}[keyof Spec] extends true
  ? true
  : false;
