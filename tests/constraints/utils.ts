import { numpy as np, Array as JaxArray } from "@jax-js/jax";

type Transform = (x: JaxArray) => JaxArray;

export function numericalLogDetJacobian(transform: Transform, x: number): number {
  const eps = 1e-4;
  const xPlus = np.array([x + eps]);
  const xMinus = np.array([x - eps]);
  const yPlus = transform(xPlus);
  const yMinus = transform(xMinus);
  const deriv = yPlus.sub(yMinus).div(2 * eps);
  return np.log(np.absolute(deriv)).item();
}
