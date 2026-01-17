import { numpy as np, Array as JaxArray } from "@jax-js/jax";

type Tolerance = { abs: number; rel: number };

export const DEFAULT_TOLERANCE: Tolerance = { abs: 0.5, rel: 0.3 };

export function drawMean(draws: JaxArray): number {
  return np.mean(draws.ref).item();
}

export function expectMeanClose(
  label: string,
  value: number,
  reference: number,
  tol: Tolerance = DEFAULT_TOLERANCE,
): void {
  const allowed = Math.max(tol.abs, Math.abs(reference) * tol.rel);
  const delta = Math.abs(value - reference);
  if (delta > allowed) {
    throw new Error(
      `${label} mean off by ${delta} (allowed ${allowed}, ref ${reference})`,
    );
  }
}
