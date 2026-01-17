let plotModule: typeof import("@observablehq/plot") | null = null;

try {
  plotModule = await import("@observablehq/plot");
} catch {
  plotModule = null;
}

export function getPlot() {
  if (!plotModule) {
    throw new Error(
      "jax-js-bayes/viz requires @observablehq/plot. Install with: npm i @observablehq/plot",
    );
  }
  return plotModule;
}
