import { Array as JaxArray } from "@jax-js/jax";
import { getPlot } from "./plot";

type Draws = Record<string, JaxArray | number[] | number[][]>;

type PairPoint = { x: number; y: number };

function flattenDraws(draws: JaxArray | number[] | number[][]): number[] {
  const jsDraws = draws instanceof JaxArray ? (draws.js() as number[] | number[][]) : draws;
  if (Array.isArray(jsDraws[0])) {
    return (jsDraws as number[][]).flat();
  }
  return jsDraws as number[];
}

export function pairPlot(
  draws: Draws,
  opts: { params?: [string, string]; width?: number; height?: number } = {},
) {
  const Plot = getPlot();
  const keys = opts.params ?? (Object.keys(draws).slice(0, 2) as [string, string]);
  if (keys.length < 2) {
    throw new Error("pairPlot requires at least two parameters");
  }

  const xVals = flattenDraws(draws[keys[0]]);
  const yVals = flattenDraws(draws[keys[1]]);
  const size = Math.min(xVals.length, yVals.length);
  const points: PairPoint[] = [];

  for (let i = 0; i < size; i++) {
    points.push({ x: xVals[i], y: yVals[i] });
  }

  return Plot.plot({
    width: opts.width,
    height: opts.height,
    marks: [Plot.dot(points, { x: "x", y: "y", opacity: 0.4 })],
  });
}
