import { Array, numpy as np } from "@jax-js/jax";

export type ShapeSpec = number | string | ReadonlyArray<number | string>;

export type DimMap = Record<string, number>;

export type InputValue = Array | number | boolean | InputValue[];

export function normalizeShapeSpec(
  shape?: ShapeSpec,
): ReadonlyArray<number | string> | undefined {
  if (shape === undefined) return undefined;
  if (typeof shape === "number" || typeof shape === "string") {
    return [shape];
  }
  return shape;
}

export function resolveShape(shape?: ShapeSpec, dims: DimMap = {}): number[] | undefined {
  const normalized = normalizeShapeSpec(shape);
  if (!normalized) return undefined;
  return normalized.map((dim) => {
    if (typeof dim === "number") return dim;
    const value = dims[dim];
    if (value === undefined) {
      throw new Error(`Unknown dimension: ${dim}`);
    }
    return value;
  });
}

export function applyShapeSpec(
  shape: ShapeSpec | undefined,
  actual: number[],
  dims: DimMap,
  label: string,
) {
  const normalized = normalizeShapeSpec(shape);
  if (!normalized) return;
  if (normalized.length !== actual.length) {
    throw new Error(
      `${label} has shape ${JSON.stringify(actual)} but expected ${JSON.stringify(normalized)}`,
    );
  }
  normalized.forEach((dim, idx) => {
    const actualDim = actual[idx];
    if (typeof dim === "number") {
      if (dim !== actualDim) {
        throw new Error(
          `${label} has shape ${JSON.stringify(actual)} but expected ${JSON.stringify(normalized)}`,
        );
      }
      return;
    }
    const existing = dims[dim];
    if (existing !== undefined && existing !== actualDim) {
      throw new Error(
        `Dimension ${dim} is ${existing} for ${label}, but got ${actualDim}`,
      );
    }
    dims[dim] = actualDim;
  });
}

export function shapesEqual(a: number[], b: number[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) return false;
  }
  return true;
}

export function coerceArray(value: InputValue): Array {
  if (value instanceof Array) {
    return value;
  }
  return np.array(value);
}

export function coerceArrayRef(value: InputValue): Array {
  if (value instanceof Array) {
    return value.ref;
  }
  return np.array(value);
}

export function sumAll(value: Array): Array {
  return np.sum(value);
}
