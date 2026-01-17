import { random, Array as JaxArray } from "@jax-js/jax";
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

export type ModelSpec = Record<string, ParamNode | DataNode | ObservedNode | DerivedNode | ((ctx: ModelContext) => JaxArray)>;

export type BoundKind = "complete" | "predictive";

export type BoundModel<S extends BoundKind = BoundKind> = {
  kind: S;
  logProb: (params: Record<string, unknown>) => JaxArray;
  samplePrior: (opts: { key: JaxArray }) => Record<string, JaxArray>;
  simulate: (params: Record<string, unknown>, opts: { key: JaxArray }) => Record<string, JaxArray>;
};

export type Model = {
  bind: (data: Record<string, unknown>) => BoundModel;
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
  const keys = random.split(key, names.length) as JaxArray[];
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
  const keys = random.split(key, observedNames.length) as JaxArray[];
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

export function model(spec: ModelSpec): Model {
  const { def, dataNodes, observedNodes, params } = buildModelDefinition(spec);

  function bind(dataInput: Record<string, unknown>): BoundModel {
    const dataValues: Record<string, JaxArray> = {};
    const observedValues: Record<string, JaxArray> = {};

    for (const [name, node] of Object.entries(dataNodes)) {
      if (!(name in dataInput)) {
        throw new Error(`Missing data '${name}'`);
      }
      const value = toArray(dataInput[name]);
      dataValues[name] = value;
    }

    for (const name of Object.keys(observedNodes)) {
      if (name in dataInput) {
        observedValues[name] = toArray(dataInput[name]);
      }
    }

    const dims = buildDimsFromData(dataNodes, dataValues);
    const boundData = { ...dataValues, ...observedValues };
    const provided = new Set(Object.keys(observedValues));

    const logProb = compileLogProb(def, boundData, provided);
    const kind: BoundKind =
      Object.keys(observedNodes).length === provided.size ? "complete" : "predictive";

    return {
      kind,
      logProb,
      samplePrior: ({ key }) => samplePriorInternal(params, dims, key),
      simulate: (paramsInput, opts) => simulateInternal(def, paramsInput, dataValues, opts.key),
    };
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
