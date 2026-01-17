import { Array as JaxArray, random } from "@jax-js/jax";
import type { Distribution } from "../types";
import { getPlot } from "./plot";

type Draws = JaxArray | number[] | number[][];

export function densityPlot(
  draws: Draws,
  opts: { width?: number; height?: number; prior?: Distribution; priorKey?: JaxArray } = {},
) {
  const Plot = getPlot();
  const jsDraws = draws instanceof JaxArray ? (draws.js() as number[] | number[][]) : draws;
  const samples = Array.isArray(jsDraws[0]) ? (jsDraws as number[][]).flat() : (jsDraws as number[]);

  const marks: any[] = [
    Plot.rectY(samples, Plot.binX({ y: "count" }, { x: (d: number) => d })),
  ];

  if (opts.prior) {
    const key = opts.priorKey ?? random.key(0);
    const priorDraws = opts.prior.sample(key, [1024]).js() as number[];
    marks.push(
      Plot.line(priorDraws, Plot.binX({ y: "count" }, { x: (d: number) => d, stroke: "orange" })),
    );
  }

  return Plot.plot({
    width: opts.width,
    height: opts.height,
    marks,
  });
}
