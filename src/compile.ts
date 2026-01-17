import { numpy as np, random, tree, type Array, type JsTree } from "@jax-js/jax";
import {
  PARAM,
  DATA,
  OBSERVED,
  type ModelSpec,
  type ParamSpec,
  type DataSpec,
  type ObservedSpec,
  type DerivedSpec,
  type BoundModel,
} from "./types";

function isParamSpec(spec: unknown): spec is ParamSpec {
  return (
    typeof spec === "object" &&
    spec !== null &&
    "kind" in spec &&
    spec.kind === PARAM
  );
}

function isDataSpec(spec: unknown): spec is DataSpec {
  return (
    typeof spec === "object" &&
    spec !== null &&
    "kind" in spec &&
    spec.kind === DATA
  );
}

function isObservedSpec(spec: unknown): spec is ObservedSpec {
  return (
    typeof spec === "object" &&
    spec !== null &&
    "kind" in spec &&
    spec.kind === OBSERVED
  );
}

function isDerivedSpec(spec: unknown): spec is DerivedSpec {
  return typeof spec === "function";
}

export function compile<Spec extends ModelSpec>(
  spec: Spec,
  providedData: Record<string, Array | number[]>,
): BoundModel<"complete" | "predictive"> {
  const paramNames: string[] = [];
  const constrainedParams: Map<string, ParamSpec> = new Map();
  const dataFields: Map<string, DataSpec> = new Map();
  const observedFields: Map<string, ObservedSpec> = new Map();
  const derivedFields: Map<string, DerivedSpec> = new Map();

  for (const [name, fieldSpec] of Object.entries(spec)) {
    if (isParamSpec(fieldSpec)) {
      paramNames.push(name);
      if (fieldSpec.constraint) {
        constrainedParams.set(name, fieldSpec);
      }
    } else if (isDataSpec(fieldSpec)) {
      dataFields.set(name, fieldSpec);
    } else if (isObservedSpec(fieldSpec)) {
      observedFields.set(name, fieldSpec);
    } else if (isDerivedSpec(fieldSpec)) {
      derivedFields.set(name, fieldSpec);
    }
  }

  const hasAllObserved = Array.from(observedFields.keys()).every(
    (name) => name in providedData,
  );
  const state = hasAllObserved ? "complete" : "predictive";

  const boundData: Record<string, Array> = {};
  for (const [name, value] of Object.entries(providedData)) {
    boundData[name] = Array.isArray(value) ? np.array(value) : value;
  }

  function logProb(params: JsTree<Array>): Array {
    const flatParams = params as Record<string, Array>;
    const context: Record<string, Array> = { ...boundData };

    let totalLogProb = np.array(0);

    for (const name of paramNames) {
      const fieldSpec = spec[name] as ParamSpec;
      let paramValue = flatParams[name];

      if (fieldSpec.constraint) {
        const unconstrained = paramValue;
        paramValue = fieldSpec.constraint.transform(unconstrained.ref);
        const ldj = fieldSpec.constraint.logDetJacobian(unconstrained);
        totalLogProb = totalLogProb.add(ldj);
      }

      context[name] = paramValue;

      const priorLogProb = fieldSpec.prior.logProb(paramValue.ref);
      totalLogProb = totalLogProb.add(priorLogProb.sum());
    }

    for (const [name, derivedFn] of derivedFields) {
      const derivedContext = buildContext(context);
      context[name] = derivedFn(derivedContext);
    }

    for (const [name, obsSpec] of observedFields) {
      if (name in boundData) {
        const obsContext = buildContext(context);
        const distribution = obsSpec.distribution(obsContext);
        const obsLogProb = distribution.logProb(boundData[name].ref);
        totalLogProb = totalLogProb.add(obsLogProb.sum());
      }
    }

    return totalLogProb;
  }

  function buildContext(
    source: Record<string, Array>,
  ): Record<string, Array> {
    const ctx: Record<string, Array> = {};
    for (const [k, v] of Object.entries(source)) {
      ctx[k] = v.ref;
    }
    return ctx;
  }

  function initialParams(): JsTree<Array> {
    const params: Record<string, Array> = {};
    for (const name of paramNames) {
      const fieldSpec = spec[name] as ParamSpec;
      let shape: number[] = [];

      if (typeof fieldSpec.shape === "string") {
        const dataName = fieldSpec.shape;
        if (dataName in boundData) {
          shape = [boundData[dataName].shape[0]];
        }
      } else if (Array.isArray(fieldSpec.shape)) {
        shape = fieldSpec.shape;
      }

      params[name] = np.zeros(shape);
    }
    return params;
  }

  function samplePrior(key: Array): JsTree<Array> {
    const params: Record<string, Array> = {};
    let currentKey = key;

    for (const name of paramNames) {
      const fieldSpec = spec[name] as ParamSpec;
      let shape: number[] = [];

      if (typeof fieldSpec.shape === "string") {
        const dataName = fieldSpec.shape;
        if (dataName in boundData) {
          shape = [boundData[dataName].shape[0]];
        }
      } else if (Array.isArray(fieldSpec.shape)) {
        shape = fieldSpec.shape;
      }

      const [paramKey, nextKey] = random.split(currentKey);
      currentKey = nextKey;

      // Sample from prior in constrained space
      const constrainedSample = fieldSpec.prior.sample(paramKey, shape);

      // Transform to unconstrained space if needed
      if (fieldSpec.constraint) {
        params[name] = fieldSpec.constraint.inverse(constrainedSample);
      } else {
        params[name] = constrainedSample;
      }
    }
    return params;
  }

  function simulate(params: JsTree<Array>, key: Array): Record<string, Array> {
    const flatParams = params as Record<string, Array>;
    const context: Record<string, Array> = { ...boundData };

    // Transform parameters to constrained space
    for (const name of paramNames) {
      const fieldSpec = spec[name] as ParamSpec;
      let paramValue = flatParams[name];

      if (fieldSpec.constraint) {
        paramValue = fieldSpec.constraint.transform(paramValue.ref);
      }

      context[name] = paramValue;
    }

    // Compute derived quantities
    for (const [name, derivedFn] of derivedFields) {
      const derivedContext = buildContext(context);
      context[name] = derivedFn(derivedContext);
    }

    // Sample from observed distributions
    const simulated: Record<string, Array> = {};
    let currentKey = key;

    for (const [name, obsSpec] of observedFields) {
      const [obsKey, nextKey] = random.split(currentKey);
      currentKey = nextKey;

      const obsContext = buildContext(context);
      const distribution = obsSpec.distribution(obsContext);

      // Determine shape from bound data if available
      let shape: number[] = [];
      if (name in boundData) {
        shape = boundData[name].shape;
      }

      simulated[name] = distribution.sample(obsKey, shape);
    }

    return simulated;
  }

  return {
    state: state as "complete" | "predictive",
    logProb,
    initialParams,
    paramNames,
    samplePrior,
    simulate,
  };
}
