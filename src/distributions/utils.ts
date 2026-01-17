import { Array as JaxArray, numpy as np } from "@jax-js/jax";

type ArrayLike = JaxArray | number | boolean | number[] | boolean[];

export function asArray(value: ArrayLike): JaxArray {
  if (value instanceof JaxArray) {
    return value;
  }
  return np.array(value);
}

export function resolveSampleShape(
  shape: number[] | undefined,
  ...params: JaxArray[]
): number[] {
  const paramShapes = params.map((param) => param.shape);
  if (shape && shape.length > 0) {
    return np.broadcastShapes(shape, ...paramShapes);
  }
  if (paramShapes.length === 0) return [];
  return np.broadcastShapes(...paramShapes);
}
