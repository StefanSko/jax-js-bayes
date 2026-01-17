import type { Array } from "@jax-js/jax";
import type { Distribution } from "./distributions/types";
import type { Constraint } from "./constraints/types";
import {
  PARAM,
  DATA,
  OBSERVED,
  type ParamSpec,
  type DataSpec,
  type ObservedSpec,
  type ModelSpec,
  type Model,
} from "./types";
import { compile } from "./compile";

export function param(
  prior: Distribution,
  options?: { constraint?: Constraint; shape?: number[] | string },
): ParamSpec {
  return {
    kind: PARAM,
    prior,
    constraint: options?.constraint,
    shape: options?.shape,
  };
}

export function data(options?: { shape?: number[] | string }): DataSpec {
  return {
    kind: DATA,
    shape: options?.shape,
  };
}

export function observed(
  distribution: (context: Record<string, Array>) => Distribution,
): ObservedSpec {
  return {
    kind: OBSERVED,
    distribution,
  };
}

export function model<Spec extends ModelSpec>(spec: Spec): Model<Spec> {
  return {
    spec,
    bind(providedData) {
      return compile(spec, providedData);
    },
  };
}
