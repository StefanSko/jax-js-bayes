import type { Array } from "@jax-js/jax";
import type { InputValue, ShapeSpec } from "./internal";

export interface Distribution {
  logProb(x: Array): Array;
  sample(key: Array, shape?: number[]): Array;
}

export interface Constraint {
  transform(unconstrained: Array): Array;
  inverse(constrained: Array): Array;
  logDetJacobian(unconstrained: Array): Array;
}

export type ParamOptions = {
  constraint?: Constraint;
  shape?: ShapeSpec;
};

export type ParamDef = {
  __kind: "param";
  distribution: Distribution;
  constraint?: Constraint;
  shape?: ShapeSpec;
};

export type DataDef = {
  __kind: "data";
  shape?: ShapeSpec;
};

export type ObservedDef = {
  __kind: "observed";
  build: (ctx: Record<string, Array>) => Distribution;
};

export type DerivedDef = (ctx: Record<string, Array>) => Array | number;

export type ModelSpec = Record<string, ParamDef | DataDef | ObservedDef | DerivedDef>;

export type ParamKeys<Spec extends ModelSpec> = {
  [K in keyof Spec]: Spec[K] extends ParamDef ? K : never;
}[keyof Spec];

export type DataKeys<Spec extends ModelSpec> = {
  [K in keyof Spec]: Spec[K] extends DataDef ? K : never;
}[keyof Spec];

export type ObservedKeys<Spec extends ModelSpec> = {
  [K in keyof Spec]: Spec[K] extends ObservedDef ? K : never;
}[keyof Spec];

export type DerivedKeys<Spec extends ModelSpec> = {
  [K in keyof Spec]: Spec[K] extends DerivedDef ? K : never;
}[keyof Spec];

export type ParamInput<Spec extends ModelSpec> = Record<ParamKeys<Spec>, InputValue>;

export type DataInput<Spec extends ModelSpec> = Record<DataKeys<Spec>, InputValue>;

export type ObservedInput<Spec extends ModelSpec> = Record<ObservedKeys<Spec>, InputValue>;

export type ParamValues<Spec extends ModelSpec> = Record<ParamKeys<Spec>, Array>;

export type DataValues<Spec extends ModelSpec> = Record<DataKeys<Spec>, Array>;

export type ObservedValues<Spec extends ModelSpec> = Record<ObservedKeys<Spec>, Array>;

export type BoundState = "complete" | "predictive";

export type BoundModel<S extends BoundState, Spec extends ModelSpec> = {
  state: S;
  data: DataValues<Spec>;
  logProb: S extends "complete" ? (params: ParamInput<Spec>) => Array : never;
};
