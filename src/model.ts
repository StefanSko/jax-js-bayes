import { numpy as np, random, Array as JaxArray } from "@jax-js/jax";
import type {
  Constraint,
  Distribution,
  DistributionFactory,
  ModelContext,
  ShapeSpec,
} from "./types";
import { applyShapeToDims, resolveShape, toArray } from "./utils";
import { compileLogProb, type ModelDefinition, type ParamDef } from "./compile";

export type ParamOptions = {
  constraint?: Constraint;
  shape?: ShapeSpec;
};

export type DataOptions = {
  shape?: ShapeSpec;
};

export type ParamNode = ParamDef & { shape?: ShapeSpec };

export type DataNode = {
  kind: "data";
  shape?: ShapeSpec;
};

export type ObservedNode = {
  kind: "observed";
  dist: Distribution | DistributionFactory;
};

export type DerivedNode = {
  kind: "derived";
  fn: (ctx: ModelContext) => JaxArray;
};

export type ModelSpec = Record<
  string,
  | ParamNode
  | DataNode
  | ObservedNode
  | DerivedNode
  | ((ctx: ModelContext) => JaxArray)
>;

export type BoundKind = "complete" | "predictive";

type DataKeys<Spec extends ModelSpec> = {
  [K in keyof Spec]: Spec[K] extends DataNode ? K : never;
}[keyof Spec];

type ObservedKeys<Spec extends ModelSpec> = {
  [K in keyof Spec]: Spec[K] extends ObservedNode ? K : never;
}[keyof Spec];

type BindInput<Spec extends ModelSpec> = Record<DataKeys<Spec>, unknown> &
  Partial<Record<ObservedKeys<Spec>, unknown>>;

type HasAllObserved<Spec extends ModelSpec, D extends Record<string, unknown>> =
  [ObservedKeys<Spec>] extends [never]
    ? true
    : Exclude<ObservedKeys<Spec>, keyof D> extends never
      ? true
      : false;

export type BoundModel<S extends BoundKind> = {
  kind: S;
  logProb: S extends "complete" ? (params: Record<string, unknown>) => JaxArray : never;
  samplePrior: (opts: { key: JaxArray }) => Record<string, JaxArray>;
  simulate: (params: Record<string, unknown>, opts: { key: JaxArray }) => Record<string, JaxArray>;
};

export type Model<Spec extends ModelSpec = ModelSpec> = {
  bind: <D extends BindInput<Spec>>(
    data: D,
  ) => BoundModel<HasAllObserved<Spec, D> extends true ? "complete" : "predictive">;
  samplePrior: (opts: { key: JaxArray; data?: Record<string, unknown> }) => Record<string, JaxArray>;
  simulate: (params: Record<string, unknown>, opts: { key: JaxArray; data?: Record<string, unknown> }) => Record<string, JaxArray>;
};

export function param(dist: Distribution | DistributionFactory, opts: ParamOptions = {}): ParamNode {
  return {
    kind: "param",
    dist,
    constraint: opts.constraint,
    shape: opts.shape,
  };
}

export function data(opts: DataOptions = {}): DataNode {
  return {
    kind: "data",
    shape: opts.shape,
  };
}

export function observed(dist: Distribution | DistributionFactory): ObservedNode {
  return {
    kind: "observed",
    dist,
  };
}

function isNode(value: unknown): value is ParamNode | DataNode | ObservedNode | DerivedNode {
  return typeof value === "object" && value !== null && "kind" in value;
}

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

function splitKeys(key: JaxArray, count: number): JaxArray[] {
  if (count <= 0) return [];
  if (count === 1) return [key];
  const keys = random.split(key, count);
  return np.split(keys, count, 0).map((part) => part.reshape([2]));
}

function buildModelDefinition(spec: ModelSpec) {
  const params: Record<string, ParamNode> = {};
  const dataNodes: Record<string, DataNode> = {};
  const observedNodes: Record<string, ObservedNode> = {};
  const derived: { name: string; fn: (ctx: ModelContext) => JaxArray }[] = [];

  for (const [name, value] of Object.entries(spec)) {
    if (typeof value === "function") {
      derived.push({ name, fn: value });
      continue;
    }
    if (!isNode(value)) {
      throw new Error(`Invalid model entry '${name}'`);
    }
    switch (value.kind) {
      case "param":
        params[name] = value;
        break;
      case "data":
        dataNodes[name] = value;
        break;
      case "observed":
        observedNodes[name] = value;
        break;
      case "derived":
        derived.push({ name, fn: value.fn });
        break;
      default:
        throw new Error(`Unknown model entry '${name}'`);
    }
  }

  const def: ModelDefinition = {
    params,
    observed: observedNodes,
    derived,
    dataNames: Object.keys(dataNodes),
  };

  return { def, dataNodes, observedNodes, params, derived };
}

function buildDimsFromData(
  dataNodes: Record<string, DataNode>,
  dataValues: Record<string, JaxArray>,
): Map<string, number> {
  const dims = new Map<string, number>();
  for (const [name, node] of Object.entries(dataNodes)) {
    const value = dataValues[name];
    applyShapeToDims(node.shape, value, dims, `data.${name}`);
  }
  return dims;
}

function samplePriorInternal(
  params: Record<string, ParamNode>,
  dims: Map<string, number>,
  key: JaxArray,
): Record<string, JaxArray> {
  const names = Object.keys(params);
  const keys = splitKeys(key, names.length);
  const ctx: ModelContext = {};
  const ctxProxy = makeContextProxy(ctx);
  const samples: Record<string, JaxArray> = {};

  names.forEach((name, idx) => {
    const defn = params[name];
    const shape = resolveShape(defn.shape, dims, `param.${name}`);
    const dist =
      typeof defn.dist === "function"
        ? (defn.dist as DistributionFactory)(ctxProxy)
        : defn.dist;
    const draw = dist.sample(keys[idx], shape);
    const constrained = defn.constraint ? defn.constraint.transform(draw) : draw;
    ctx[name] = constrained;
    samples[name] = constrained;
  });

  return samples;
}

function simulateInternal(
  def: ModelDefinition,
  params: Record<string, unknown>,
  dataValues: Record<string, JaxArray>,
  key: JaxArray,
): Record<string, JaxArray> {
  const ctx: ModelContext = { ...dataValues };
  const ctxProxy = makeContextProxy(ctx);
  for (const [name, defn] of Object.entries(def.params)) {
    if (params[name] === undefined) {
      throw new Error(`simulate: missing parameter '${name}'`);
    }
    const value = toArray(params[name]);
    ctx[name] = value;
  }
  for (const derived of def.derived) {
    ctx[derived.name] = derived.fn(ctxProxy);
  }

  const observedNames = Object.keys(def.observed);
  const keys = splitKeys(key, observedNames.length);
  const result: Record<string, JaxArray> = { ...dataValues };

  observedNames.forEach((name, idx) => {
    const defn = def.observed[name];
    const dist =
      typeof defn.dist === "function"
        ? (defn.dist as DistributionFactory)(ctxProxy)
        : defn.dist;
    result[name] = dist.sample(keys[idx]);
  });

  return result;
}

export function model<Spec extends ModelSpec>(spec: Spec): Model<Spec> {
  const { def, dataNodes, observedNodes, params } = buildModelDefinition(spec);

  function bind<D extends BindInput<Spec>>(
    dataInput: D,
  ): BoundModel<HasAllObserved<Spec, D> extends true ? "complete" : "predictive"> {
    const dataLookup = dataInput as Record<string, unknown>;
    const dataValues: Record<string, JaxArray> = {};
    const observedValues: Record<string, JaxArray> = {};

    for (const [name, node] of Object.entries(dataNodes)) {
      if (!(name in dataLookup)) {
        throw new Error(`Missing data '${name}'`);
      }
      const value = toArray(dataLookup[name]);
      dataValues[name] = value;
    }

    const observedNames = Object.keys(observedNodes);
    const providedObserved = observedNames.filter((name) => name in dataLookup);
    if (providedObserved.length > 0 && providedObserved.length < observedNames.length) {
      throw new Error(
        "Partial observed data provided. Supply all observed values for a complete model, or none for predictive.",
      );
    }
    for (const name of providedObserved) {
      observedValues[name] = toArray(dataLookup[name]);
    }

    const dims = buildDimsFromData(dataNodes, dataValues);
    const boundData = { ...dataValues, ...observedValues };
    const kind: BoundKind =
      observedNames.length === providedObserved.length ? "complete" : "predictive";

    if (kind === "complete") {
      const logProb = compileLogProb(def, boundData, new Set(providedObserved));
      const complete = {
        kind,
        logProb,
        samplePrior: ({ key }) => samplePriorInternal(params, dims, key),
        simulate: (paramsInput, opts) => simulateInternal(def, paramsInput, dataValues, opts.key),
      } as BoundModel<"complete">;
      return complete as BoundModel<
        HasAllObserved<Spec, D> extends true ? "complete" : "predictive"
      >;
    }

    const predictive = {
      kind,
      samplePrior: ({ key }) => samplePriorInternal(params, dims, key),
      simulate: (paramsInput, opts) => simulateInternal(def, paramsInput, dataValues, opts.key),
    } as BoundModel<"predictive">;
    return predictive as BoundModel<
      HasAllObserved<Spec, D> extends true ? "complete" : "predictive"
    >;
  }

  function samplePrior({ key, data: dataInput = {} }: { key: JaxArray; data?: Record<string, unknown> }) {
    const dataValues: Record<string, JaxArray> = {};
    for (const [name, node] of Object.entries(dataNodes)) {
      if (!(name in dataInput)) {
        continue;
      }
      const value = toArray(dataInput[name]);
      dataValues[name] = value;
    }
    const dims = buildDimsFromData(dataNodes, dataValues);
    return samplePriorInternal(params, dims, key);
  }

  function simulate(
    paramsInput: Record<string, unknown>,
    opts: { key: JaxArray; data?: Record<string, unknown> },
  ) {
    const dataValues: Record<string, JaxArray> = {};
    const dataInput = opts.data ?? {};
    for (const [name] of Object.entries(dataNodes)) {
      if (!(name in dataInput)) {
        throw new Error(`simulate: missing data '${name}'`);
      }
      dataValues[name] = toArray(dataInput[name]);
    }
    return simulateInternal(def, paramsInput, dataValues, opts.key);
  }

  return { bind, samplePrior, simulate };
}
