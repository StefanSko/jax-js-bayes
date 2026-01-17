import { getPlot, toJsValues, flattenValues } from "./utils";

export function densityPlot(draws: unknown): HTMLElement {
  const Plot = getPlot();
  const values = flattenValues(toJsValues(draws));
  return Plot.plot({
    marks: [Plot.rectY(values, Plot.binX({ y: "count" }))],
  });
}
