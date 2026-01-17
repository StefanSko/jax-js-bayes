import { getPlot, toJsValues, flattenValues } from "./utils";

export function tracePlot(draws: unknown): HTMLElement {
  const Plot = getPlot();
  const values = flattenValues(toJsValues(draws));
  return Plot.plot({
    marks: [Plot.lineY(values)],
  });
}
