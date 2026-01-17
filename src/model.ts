import { Array as JaxArray, numpy as np, random } from "@jax-js/jax";
import {
  applyShapeSpec,
  coerceArray,
  coerceArrayRef,
  DimMap,
  InputValue,
  resolveShape,
  shapesEqual,
  sumAll,
} from "./internal";
import type {
  BoundModel,
  BoundState,
  DataDef,
  DataInput,
  DataValues,
  DerivedDef,
  Distribution,
  ModelSpec,
  ObservedDef,
  ObservedInput,
  ObservedValues,
  ParamDef,
  ParamInput,
  ParamValues,
  ParamOptions,
} from "./types";

const PARAM_KIND = "param" as const;
const DATA_KIND = "data" as const;
const OBSERVED_KIND = "observed" as const;

export function param(distribution: Distribution, opts: ParamOptions = {}): ParamDef {
  return {
    __kind: PARAM_KIND,
    distribution,
    constraint: opts.constraint,
    shape: opts.shape,
  };
}

export function data(opts: { shape?: ParamOptions["shape"] } = {}): DataDef {
  return {
    __kind: DATA_KIND,
    shape: opts.shape,
  };
}

export function observed(build: ObservedDef["build"]): ObservedDef {
  return {
    __kind: OBSERVED_KIND,
    build,
  };
}

type ModelEntry =
  | { kind: typeof PARAM_KIND; key: string; def: ParamDef }
  | { kind: typeof DATA_KIND; key: string; def: DataDef }
  | { kind: typeof OBSERVED_KIND; key: string; def: ObservedDef }
  | { kind: "derived"; key: string; def: DerivedDef };

function isParamDef(value: unknown): value is ParamDef {
  return Boolean(value) && (value as ParamDef).__kind === PARAM_KIND;
}

function isDataDef(value: unknown): value is DataDef {
  return Boolean(value) && (value as DataDef).__kind === DATA_KIND;
}

function isObservedDef(value: unknown): value is ObservedDef {
  return Boolean(value) && (value as ObservedDef).__kind === OBSERVED_KIND;
}

function splitKeys(key: JaxArray, count: number): JaxArray[] {
  if (count <= 0) return [];
  if (count === 1) return [key];
  const keys = random.split(key.ref, count);
  const parts = np.split(keys, count, 0);
  return parts.map((part) => np.squeeze(part, 0));
}

function contextProxy(context: Record<string, JaxArray>): Record<string, JaxArray> {
  return new Proxy(context, {
    get(target, prop) {
      if (typeof prop !== "string") return undefined;
      const value = target[prop];
      if (value instanceof JaxArray) {
        return value.ref;
      }
      return value;
    },
  }) as Record<string, JaxArray>;
}

export function model<Spec extends ModelSpec>(spec: Spec) {
  const entries: ModelEntry[] = [];
  const paramEntries: Array<{ key: string; def: ParamDef }> = [];
  const dataEntries: Array<{ key: string; def: DataDef }> = [];
  const observedEntries: Array<{ key: string; def: ObservedDef }> = [];
  const derivedEntries: Array<{ key: string; def: DerivedDef }> = [];

  for (const [key, value] of Object.entries(spec)) {
    if (isParamDef(value)) {
      entries.push({ kind: PARAM_KIND, key, def: value });
      paramEntries.push({ key, def: value });
    } else if (isDataDef(value)) {
      entries.push({ kind: DATA_KIND, key, def: value });
      dataEntries.push({ key, def: value });
    } else if (isObservedDef(value)) {
      entries.push({ kind: OBSERVED_KIND, key, def: value });
      observedEntries.push({ key, def: value });
    } else if (typeof value === "function") {
      entries.push({ kind: "derived", key, def: value as DerivedDef });
      derivedEntries.push({ key, def: value as DerivedDef });
    } else {
      throw new Error(`Unsupported model entry for ${key}`);
    }
  }

  function bind(
    input: DataInput<Spec> & Partial<ObservedInput<Spec>>,
    opts: { dims?: DimMap } = {},
  ): BoundModel<BoundState, Spec> {
    const dims: DimMap = { ...opts.dims };
    const allowedKeys = new Set<string>([
      ...dataEntries.map((entry) => entry.key),
      ...observedEntries.map((entry) => entry.key),
    ]);

    for (const key of Object.keys(input)) {
      if (!allowedKeys.has(key)) {
        throw new Error(`Unknown data field: ${key}`);
      }
    }

    const boundData: Record<string, JaxArray> = {};
    for (const { key, def } of dataEntries) {
      if (!(key in input)) {
        throw new Error(`Missing data field: ${key}`);
      }
      const arr = coerceArray(input[key] as InputValue);
      applyShapeSpec(def.shape, arr.shape, dims, `data.${key}`);
      boundData[key] = arr;
    }

    const providedObserved = observedEntries.filter((entry) => entry.key in input);
    if (providedObserved.length > 0 && providedObserved.length !== observedEntries.length) {
      throw new Error(
        "Either provide all observed values for a complete model, or none for predictive",
      );
    }

    const boundObserved: Record<string, JaxArray> = {};
    if (providedObserved.length === observedEntries.length) {
      for (const { key } of observedEntries) {
        const arr = coerceArray(input[key] as InputValue);
        boundObserved[key] = arr;
      }
    }

    const paramShapes = new Map<string, number[] | undefined>();
    for (const { key, def } of paramEntries) {
      paramShapes.set(key, resolveShape(def.shape, dims));
    }

    const state: BoundState =
      providedObserved.length === observedEntries.length ? "complete" : "predictive";

    const bound: any = {
      state,
      data: boundData,
    };

    if (state === "complete") {
      bound.logProb = compileLogProb(
        paramEntries,
        derivedEntries,
        observedEntries,
        boundData,
        boundObserved,
        paramShapes,
      );
    }

    return bound as BoundModel<BoundState, Spec>;
  }

  function compileLogProb(
    params: Array<{ key: string; def: ParamDef }>,
    derived: Array<{ key: string; def: DerivedDef }>,
    observedDefs: Array<{ key: string; def: ObservedDef }>,
    dataValues: Record<string, JaxArray>,
    observedValues: Record<string, JaxArray>,
    paramShapes: Map<string, number[] | undefined>,
  ) {
    return (inputParams: ParamInput<Spec>): JaxArray => {
      const owned: JaxArray[] = [];
      const ctx: Record<string, JaxArray> = {
        ...dataValues,
      };

      let total = np.array(0);

      for (const { key, def } of params) {
        if (!(key in inputParams)) {
          throw new Error(`Missing parameter: ${key}`);
        }
        const value = coerceArrayRef(inputParams[key] as InputValue);
        const expectedShape = paramShapes.get(key);
        if (expectedShape && !shapesEqual(value.shape, expectedShape)) {
          throw new Error(
            `Parameter ${key} has shape ${JSON.stringify(value.shape)}, expected ${JSON.stringify(expectedShape)}`,
          );
        }
        owned.push(value);

        if (def.constraint) {
          const constrained = def.constraint.transform(value.ref);
          owned.push(constrained);
          const logDet = def.constraint.logDetJacobian(value.ref);
          const logProb = def.distribution.logProb(constrained.ref);
          total = total.add(sumAll(logProb));
          total = total.add(sumAll(logDet));
          ctx[key] = constrained;
        } else {
          const logProb = def.distribution.logProb(value.ref);
          total = total.add(sumAll(logProb));
          ctx[key] = value;
        }
      }

      for (const { key, def } of derived) {
        const value = def(contextProxy(ctx));
        const arr = value instanceof JaxArray ? value : np.array(value);
        owned.push(arr);
        ctx[key] = arr;
      }

      for (const { key, def } of observedDefs) {
        const dist = def.build(contextProxy(ctx));
        const obs = observedValues[key];
        const logProb = dist.logProb(obs.ref);
        total = total.add(sumAll(logProb));
      }

      for (const arr of owned) {
        arr.dispose();
      }

      return total;
    };
  }

  function samplePrior(
    opts: { key: JaxArray; dims?: DimMap } = { key: random.key(0) },
  ) {
    const dims = { ...opts.dims };
    const output: Record<string, JaxArray> = {};
    const keys = splitKeys(opts.key, paramEntries.length);

    paramEntries.forEach(({ key, def }, idx) => {
      const shape = resolveShape(def.shape, dims);
      output[key] = def.distribution.sample(keys[idx], shape ?? []);
    });

    return output as ParamValues<Spec>;
  }

  function simulate(
    params: ParamInput<Spec>,
    opts: { key: JaxArray; data?: DataInput<Spec>; dims?: DimMap },
  ) {
    const dims = { ...opts.dims };
    const dataInput = opts.data ?? ({} as DataInput<Spec>);
    const ctx: Record<string, JaxArray> = {};
    const owned: JaxArray[] = [];

    for (const { key, def } of dataEntries) {
      if (!(key in dataInput)) {
        throw new Error(`Missing data field for simulation: ${key}`);
      }
      const arr = coerceArray(dataInput[key] as InputValue);
      applyShapeSpec(def.shape, arr.shape, dims, `data.${key}`);
      ctx[key] = arr;
    }

    for (const { key, def } of paramEntries) {
      if (!(key in params)) {
        throw new Error(`Missing parameter for simulation: ${key}`);
      }
      const arr = coerceArrayRef(params[key] as InputValue);
      const expectedShape = resolveShape(def.shape, dims);
      if (expectedShape && !shapesEqual(arr.shape, expectedShape)) {
        throw new Error(
          `Parameter ${key} has shape ${JSON.stringify(arr.shape)}, expected ${JSON.stringify(expectedShape)}`,
        );
      }
      ctx[key] = arr;
      owned.push(arr);
    }

    for (const { key, def } of derivedEntries) {
      const value = def(contextProxy(ctx));
      const arr = value instanceof JaxArray ? value : np.array(value);
      ctx[key] = arr;
      owned.push(arr);
    }

    const keys = splitKeys(opts.key, observedEntries.length);
    const output: Record<string, JaxArray> = {};

    observedEntries.forEach(({ key, def }, idx) => {
      const dist = def.build(contextProxy(ctx));
      output[key] = dist.sample(keys[idx]);
    });

    for (const { key } of dataEntries) {
      output[key] = ctx[key];
    }

    for (const arr of owned) {
      arr.dispose();
    }

    return output as DataValues<Spec> & ObservedValues<Spec>;
  }

  return {
    bind,
    samplePrior,
    simulate,
  };
}
