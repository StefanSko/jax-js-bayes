import { numpy as np, Array as JaxArray } from "@jax-js/jax";
import type { Constraint, Distribution, DistributionFactory, ModelContext } from "./types";
import { sumToScalar, toArray } from "./utils";

export type ParamDef = {
  kind: "param";
  dist: Distribution | DistributionFactory;
  constraint?: Constraint;
  shape?: unknown;
};

export type ObservedDef = {
  kind: "observed";
  dist: Distribution | DistributionFactory;
};

export type DerivedDef = {
  name: string;
  fn: (ctx: ModelContext) => JaxArray;
};

export type ModelDefinition = {
  params: Record<string, ParamDef>;
  observed: Record<string, ObservedDef>;
  derived: DerivedDef[];
  dataNames: string[];
};

export type BoundData = Record<string, JaxArray>;

function makeContextProxy(ctx: ModelContext): ModelContext {
  return new Proxy(ctx, {
    get(target, prop) {
      const value = target[prop as keyof ModelContext];
      if (value && typeof value === "object" && "ref" in value) {
        return (value as JaxArray).ref;
      }
      return value as any;
    },
  });
}

export function compileLogProb(
  def: ModelDefinition,
  data: BoundData,
  observedProvided: Set<string>,
): (params: Record<string, unknown>) => JaxArray {
  const paramNames = Object.keys(def.params);
  const observedNames = Object.keys(def.observed).filter((name) =>
    observedProvided.has(name),
  );

  return (params: Record<string, unknown>) => {
    let logProb = np.array(0);
    const ctx: ModelContext = {};
    const ctxProxy = makeContextProxy(ctx);

    for (const dataName of def.dataNames) {
      ctx[dataName] = data[dataName];
    }
    for (const obsName of observedNames) {
      ctx[obsName] = data[obsName];
    }

    for (const name of paramNames) {
      const defn = def.params[name];
      if (params[name] === undefined) {
        throw new Error(`Missing parameter '${name}'`);
      }
      const raw = toArray(params[name]);
      const rawRef = raw.ref;
      const constrained = defn.constraint
        ? defn.constraint.transform(raw)
        : raw;
      ctx[name] = constrained;
      if (defn.constraint) {
        logProb = logProb.add(
          sumToScalar(defn.constraint.logDetJacobian(rawRef)),
        );
      }
    }

    for (const derived of def.derived) {
      ctx[derived.name] = derived.fn(ctxProxy);
    }

    for (const name of paramNames) {
      const defn = def.params[name];
      const dist =
        typeof defn.dist === "function"
          ? (defn.dist as DistributionFactory)(ctxProxy)
          : defn.dist;
      logProb = logProb.add(sumToScalar(dist.logProb(ctxProxy[name])));
    }

    for (const name of observedNames) {
      const defn = def.observed[name];
      const dist =
        typeof defn.dist === "function"
          ? (defn.dist as DistributionFactory)(ctxProxy)
          : defn.dist;
      logProb = logProb.add(sumToScalar(dist.logProb(ctxProxy[name])));
    }

    return logProb;
  };
}
