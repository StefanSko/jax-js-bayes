import { Array as JaxArray } from "@jax-js/jax";
import { getPlot } from "./plot";

type Draws = JaxArray | number[] | number[][];

type TracePoint = { sample: number; value: number; chain?: number };

export function tracePlot(draws: Draws, opts: { width?: number; height?: number } = {}) {
  const Plot = getPlot();
  const jsDraws = draws instanceof JaxArray ? (draws.js() as number[] | number[][]) : draws;
  const series: TracePoint[] = [];

  if (Array.isArray(jsDraws[0])) {
    const chains = jsDraws as number[][];
    chains.forEach((chain, chainIdx) => {
      chain.forEach((value, sample) => {
        series.push({ sample, value, chain: chainIdx });
      });
    });
  } else {
    const samples = jsDraws as number[];
    samples.forEach((value, sample) => {
      series.push({ sample, value });
    });
  }

  return Plot.plot({
    width: opts.width,
    height: opts.height,
    marks: [
      Plot.line(series, {
        x: "sample",
        y: "value",
        stroke: "chain",
      }),
    ],
  });
}
