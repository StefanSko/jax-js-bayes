import { tree } from "@jax-js/jax";

export const memLogEnabled = process.env.POSTERIORDB_MEM_LOG === "1";
export const memGcEnabled = process.env.POSTERIORDB_MEM_GC === "1";
export const memDisposeEnabled = process.env.POSTERIORDB_MEM_DISPOSE === "1";

function formatMB(bytes: number): string {
  return `${(bytes / 1024 / 1024).toFixed(1)}MB`;
}

export function logMemory(label: string): void {
  if (!memLogEnabled) return;
  if (memGcEnabled && typeof global.gc === "function") {
    global.gc();
  }
  const { heapUsed, heapTotal, rss, external, arrayBuffers } =
    process.memoryUsage();
  const parts = [
    `heap=${formatMB(heapUsed)}/${formatMB(heapTotal)}`,
    `rss=${formatMB(rss)}`,
    `ext=${formatMB(external)}`,
  ];
  if (typeof arrayBuffers === "number") {
    parts.push(`ab=${formatMB(arrayBuffers)}`);
  }
  console.log(`[mem] ${label} ${parts.join(" ")}`);
}

export function maybeDisposeTree(value: unknown, label?: string): void {
  if (!memDisposeEnabled) return;
  try {
    tree.dispose(value as never);
  } catch (err) {
    if (memLogEnabled) {
      const suffix = label ? ` (${label})` : "";
      console.warn(`[mem] dispose failed${suffix}`, err);
    }
  }
}

export function disposeHmcResult(result: {
  draws?: unknown;
  stats?: { massMatrix?: unknown };
}): void {
  if (!memDisposeEnabled) return;
  maybeDisposeTree(result.draws, "draws");
  const massMatrix = result.stats?.massMatrix;
  if (Array.isArray(massMatrix)) {
    for (const entry of massMatrix) {
      maybeDisposeTree(entry, "massMatrix");
    }
  } else if (massMatrix) {
    maybeDisposeTree(massMatrix, "massMatrix");
  }
}
