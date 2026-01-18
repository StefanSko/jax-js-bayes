// Model DSL
export { model, param, data, observed } from "./model";

// Types
export type {
  ParamSpec,
  DataSpec,
  ObservedSpec,
  DerivedSpec,
  ModelSpec,
  ModelEntry,
  Model,
  BoundModel,
} from "./types";

// Re-export distributions and constraints from submodules
// Users can import from "jax-js-bayes/distributions" and "jax-js-bayes/constraints"
