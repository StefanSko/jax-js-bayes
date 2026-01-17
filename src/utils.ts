import { Array as JaxArray, numpy as np } from "@jax-js/jax";
import type { ShapeSpec, ShapeToken } from "./types";

export function isJaxArray(value: unknown): value is JaxArray {
  return value instanceof JaxArray;
}

export function toArray(value: unknown): JaxArray {
  if (value instanceof JaxArray) {
    return value;
  }
  if (
    value &&
    typeof value === "object" &&
    "aval" in value &&
    "ref" in value
  ) {
    return value as JaxArray;
  }
  return np.array(value as number | number[] | number[][]);
}

export function normalizeShape(shape?: ShapeSpec): ShapeToken[] {
  if (shape === undefined) return [];
  if (Array.isArray(shape)) return shape;
  return [shape];
}

export function applyShapeToDims(
  shape: ShapeSpec | undefined,
  value: JaxArray,
  dims: Map<string, number>,
  label: string,
): void {
  const tokens = normalizeShape(shape);
  if (tokens.length === 0) return;
  if (value.ndim === 0 && tokens.length === 1) {
    const size = value.item();
    if (!Number.isFinite(size) || size <= 0 || !Number.isInteger(size)) {
      throw new Error(`${label}: invalid dimension size ${size}`);
    }
    const token = tokens[0];
    if (typeof token === "string") {
      const existing = dims.get(token);
      if (existing === undefined) {
        dims.set(token, size);
      } else if (existing !== size) {
        throw new Error(`${label}: dimension '${token}' expected ${existing} but got ${size}`);
      }
    } else if (token !== size) {
      throw new Error(`${label}: expected dimension ${token} but got ${size}`);
    }
    return;
  }

  const actualShape = value.shape;
  if (tokens.length !== actualShape.length) {
    throw new Error(
      `${label}: shape ${JSON.stringify(tokens)} does not match data shape ` +
        JSON.stringify(actualShape),
    );
  }

  tokens.forEach((token, idx) => {
    if (typeof token === "string") {
      const existing = dims.get(token);
      if (existing === undefined) {
        dims.set(token, actualShape[idx]);
      } else if (existing !== actualShape[idx]) {
        throw new Error(
          `${label}: dimension '${token}' expected ${existing} but got ` +
            `${actualShape[idx]}`,
        );
      }
    } else if (token !== actualShape[idx]) {
      throw new Error(
        `${label}: expected dimension ${token} but got ${actualShape[idx]}`,
      );
    }
  });
}

export function resolveShape(
  shape: ShapeSpec | undefined,
  dims: Map<string, number>,
  label: string,
): number[] {
  const tokens = normalizeShape(shape);
  return tokens.map((token) => {
    if (typeof token === "number") return token;
    const value = dims.get(token);
    if (value === undefined) {
      throw new Error(`${label}: unknown dimension '${token}'`);
    }
    return value;
  });
}

export function sumToScalar(value: JaxArray): JaxArray {
  if (value.ndim === 0) return value;
  return np.sum(value);
}
