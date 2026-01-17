import { getPlot, toJsValues } from "./utils";

export function pairPlot(
  draws: Record<string, unknown>,
  { params }: { params: [string, string] },
): HTMLElement {
  const Plot = getPlot();
  const xValues = toJsValues(draws[params[0]]) as number[] | number[][];
  const yValues = toJsValues(draws[params[1]]) as number[] | number[][];
  const x = Array.isArray(xValues[0]) ? (xValues as number[][]).flat() : (xValues as number[]);
  const y = Array.isArray(yValues[0]) ? (yValues as number[][]).flat() : (yValues as number[]);

  const points = x.map((value, idx) => ({ x: value, y: y[idx] }));

  return Plot.plot({
    marks: [Plot.dot(points, { x: "x", y: "y" })],
  });
}
