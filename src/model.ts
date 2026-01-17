import { numpy as np, random, Array, JsTree } from "@jax-js/jax";

/**
 * Split a key into multiple keys.
 */
function splitKeys(key: Array, num: number): Array[] {
  const keys = random.split(key, num);
  return np.split(keys, num, 0).map((k) => k.reshape([2]));
}
import type {
  ParamSpec,
  DataSpec,
  ObservedSpec,
  DerivedSpec,
  ModelSpec,
  ModelEntry,
  Model,
  BoundModel,
} from "./types";
import type { Distribution } from "./distributions/types";
import type { Constraint } from "./constraints/types";

/**
 * Create a parameter specification.
 */
export function param(
  prior: Distribution,
  options?: { constraint?: Constraint; shape?: string | number[] },
): ParamSpec {
  return {
    type: "param",
    prior,
    constraint: options?.constraint,
    shape: options?.shape,
  };
}

/**
 * Create a data specification.
 */
export function data(options?: { shape?: string | number[] }): DataSpec {
  return {
    type: "data",
    shape: options?.shape,
  };
}

/**
 * Create an observed (likelihood) specification.
 */
export function observed(
  likelihood: (params: Record<string, Array>) => Distribution,
): ObservedSpec {
  return {
    type: "observed",
    likelihood,
  };
}

/**
 * Check if entry is a param spec.
 */
function isParam(entry: ModelEntry): entry is ParamSpec {
  return typeof entry === "object" && "type" in entry && entry.type === "param";
}

/**
 * Check if entry is a data spec.
 */
function isData(entry: ModelEntry): entry is DataSpec {
  return typeof entry === "object" && "type" in entry && entry.type === "data";
}

/**
 * Check if entry is an observed spec.
 */
function isObserved(entry: ModelEntry): entry is ObservedSpec {
  return typeof entry === "object" && "type" in entry && entry.type === "observed";
}

/**
 * Check if entry is a derived function.
 */
function isDerived(entry: ModelEntry): entry is DerivedSpec {
  return typeof entry === "function";
}

/**
 * Create a model from a specification.
 */
export function model<Spec extends ModelSpec>(spec: Spec): Model<Spec> {
  // Extract param, data, observed, and derived names
  const paramNames: string[] = [];
  const dataNames: string[] = [];
  const observedNames: string[] = [];
  const derivedNames: string[] = [];

  for (const [name, entry] of Object.entries(spec)) {
    if (isParam(entry)) {
      paramNames.push(name);
    } else if (isData(entry)) {
      dataNames.push(name);
    } else if (isObserved(entry)) {
      observedNames.push(name);
    } else if (isDerived(entry)) {
      derivedNames.push(name);
    }
  }

  return {
    spec,

    bind<D extends Record<string, Array | number[]>>(
      inputData: D,
    ): BoundModel<"complete"> | BoundModel<"predictive"> {
      // Convert number arrays to jax-js Arrays
      const boundData: Record<string, Array> = {};
      for (const [key, value] of Object.entries(inputData)) {
        boundData[key] = value instanceof Array ? value : np.array(value as number[]);
      }

      // Check if all observed variables have data
      const hasAllObserved = observedNames.every((name) => name in boundData);
      const state = hasAllObserved ? "complete" : "predictive";

      if (state === "complete") {
        // Create the logProb function
        const logProbFn = (uncontstrainedParams: JsTree<Array>): Array => {
          const params = uncontstrainedParams as Record<string, Array>;

          // Start accumulating log probability
          let logP = np.array(0);

          // Apply constraints and add Jacobian adjustments + prior log probs
          const constrainedParams: Record<string, Array> = {};

          for (const name of paramNames) {
            const paramSpec = spec[name] as ParamSpec;
            const unconstrained = params[name];

            if (paramSpec.constraint) {
              // Apply constraint
              const constrained = paramSpec.constraint.transform(unconstrained.ref);
              constrainedParams[name] = constrained;

              // Add Jacobian adjustment
              const logDetJ = paramSpec.constraint.logDetJacobian(unconstrained.ref);
              logP = logP.add(logDetJ);

              // Add prior log prob (on constrained value)
              const priorLogP = paramSpec.prior.logProb(constrained.ref);
              logP = logP.add(priorLogP);
            } else {
              // No constraint
              constrainedParams[name] = unconstrained.ref;

              // Add prior log prob directly
              const priorLogP = paramSpec.prior.logProb(unconstrained.ref);
              logP = logP.add(priorLogP);
            }
          }

          // Add data to the params object for derived and likelihood computations
          const allValues: Record<string, Array> = {
            ...constrainedParams,
            ...boundData,
          };

          // Compute derived quantities
          for (const name of derivedNames) {
            const derivedFn = spec[name] as DerivedSpec;
            // Create a ref'd version of allValues for the derived function
            const refValues: Record<string, Array> = {};
            for (const [k, v] of Object.entries(allValues)) {
              refValues[k] = v.ref;
            }
            allValues[name] = derivedFn(refValues);
          }

          // Add likelihood log probs
          for (const name of observedNames) {
            const observedSpec = spec[name] as ObservedSpec;
            const observedData = boundData[name];

            // Create a ref'd version for the likelihood function
            const refValues: Record<string, Array> = {};
            for (const [k, v] of Object.entries(allValues)) {
              refValues[k] = v.ref;
            }

            const likelihoodDist = observedSpec.likelihood(refValues);
            const likelihoodLogP = likelihoodDist.logProb(observedData.ref);
            logP = logP.add(likelihoodLogP);
          }

          return logP;
        };

        return {
          logProb: logProbFn,
          data: boundData,
          state: "complete" as const,
        } as BoundModel<"complete">;
      } else {
        return {
          logProb: undefined as never,
          data: boundData,
          state: "predictive" as const,
        } as BoundModel<"predictive">;
      }
    },

    samplePrior(options: { key: Array; dims?: Record<string, number> }): JsTree<Array> {
      let key = options.key;
      const dims = options.dims ?? {};
      const samples: Record<string, Array> = {};

      for (const name of paramNames) {
        const paramSpec = spec[name] as ParamSpec;

        // Determine shape
        let shape: number[] = [];
        if (paramSpec.shape) {
          if (typeof paramSpec.shape === "string") {
            // Named dimension
            const dimSize = dims[paramSpec.shape];
            if (dimSize === undefined) {
              throw new Error(
                `Dimension "${paramSpec.shape}" not provided in dims for parameter "${name}"`,
              );
            }
            shape = [dimSize];
          } else {
            shape = paramSpec.shape;
          }
        }

        // Split key for this sample
        const keys = splitKeys(key, 2);
        key = keys[0];
        const sampleKey = keys[1];

        // Sample from prior
        const sample = paramSpec.prior.sample(sampleKey, shape);
        samples[name] = sample;
      }

      return samples;
    },

    simulate(
      params: JsTree<Array>,
      options: { key: Array; dims?: Record<string, number> },
    ): Record<string, Array> {
      let key = options.key;
      const dims = options.dims ?? {};
      const paramsRecord = params as Record<string, Array>;
      const simulated: Record<string, Array> = {};

      // Build all values including params and derived
      const allValues: Record<string, Array> = { ...paramsRecord };

      // Compute derived quantities
      for (const name of derivedNames) {
        const derivedFn = spec[name] as DerivedSpec;
        const refValues: Record<string, Array> = {};
        for (const [k, v] of Object.entries(allValues)) {
          refValues[k] = v.ref;
        }
        allValues[name] = derivedFn(refValues);
      }

      // Simulate observed data
      for (const name of observedNames) {
        const observedSpec = spec[name] as ObservedSpec;

        // Determine shape for this observed variable
        let shape: number[] = [];
        const dimSize = dims[name];
        if (dimSize !== undefined) {
          shape = [dimSize];
        }

        // Create refs for the likelihood
        const refValues: Record<string, Array> = {};
        for (const [k, v] of Object.entries(allValues)) {
          refValues[k] = v.ref;
        }

        const likelihoodDist = observedSpec.likelihood(refValues);

        // Split key for this sample
        const keys = splitKeys(key, 2);
        key = keys[0];
        const sampleKey = keys[1];

        // Sample from likelihood
        simulated[name] = likelihoodDist.sample(sampleKey, shape);
      }

      return simulated;
    },
  };
}
