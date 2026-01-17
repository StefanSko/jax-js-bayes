import { Array as JaxArray } from "@jax-js/jax";
import { createRequire } from "node:module";

const PLOT_ERROR =
  "jax-js-bayes/viz requires @observablehq/plot. Install with: npm i @observablehq/plot";

let cachedPlot: any | null = null;

export function getPlot(): any {
  if (cachedPlot) return cachedPlot;
  const globalPlot = (globalThis as { Plot?: unknown }).Plot;
  if (globalPlot) {
    cachedPlot = globalPlot;
    return cachedPlot;
  }
  try {
    const require = createRequire(import.meta.url);
    cachedPlot = require("@observablehq/plot");
    return cachedPlot;
  } catch {
    throw new Error(PLOT_ERROR);
  }
}

export function toJsValues(values: unknown): number[] | number[][] {
  if (values instanceof JaxArray) {
    return values.js() as number[] | number[][];
  }
  return values as number[] | number[][];
}

export function flattenValues(values: number[] | number[][]): number[] {
  if (!Array.isArray(values[0])) {
    return values as number[];
  }
  const out: number[] = [];
  for (const row of values as number[][]) {
    out.push(...row);
  }
  return out;
}
