import fs from "node:fs";
import path from "node:path";
import os from "node:os";
import { fileURLToPath } from "node:url";
import AdmZip from "adm-zip";

const DEFAULT_PDB = path.join(os.homedir(), ".posteriordb", "posterior_database");
const LOCAL_REFERENCE_DRAWS = path.join(
  path.dirname(fileURLToPath(import.meta.url)),
  "reference_draws",
);

export const POSTERIORDB_PATH =
  process.env.POSTERIORDB_PATH ?? DEFAULT_PDB;

function readZipJson(filePath: string): unknown {
  if (!fs.existsSync(filePath)) {
    throw new Error(`Missing data file: ${filePath}`);
  }
  const zip = new AdmZip(filePath);
  const entries = zip.getEntries();
  if (entries.length === 0) {
    throw new Error(`Empty zip: ${filePath}`);
  }
  const content = entries[0].getData().toString("utf-8");
  return JSON.parse(content);
}

function loadLocalMeanStats(posteriorName: string): {
  names: string[];
  mean_value: number[];
} | null {
  const localZip = path.join(LOCAL_REFERENCE_DRAWS, `${posteriorName}.json.zip`);
  if (!fs.existsSync(localZip)) {
    return null;
  }

  const draws = readZipJson(localZip) as Record<string, number[]>[];
  const sums = new Map<string, number>();
  const counts = new Map<string, number>();

  for (const chain of draws) {
    for (const [name, values] of Object.entries(chain)) {
      if (name.startsWith("lp__")) {
        continue;
      }
      let sum = sums.get(name) ?? 0;
      for (const value of values) {
        sum += value;
      }
      sums.set(name, sum);
      counts.set(name, (counts.get(name) ?? 0) + values.length);
    }
  }

  const names = Array.from(sums.keys()).sort();
  const mean_value = names.map((name) => sums.get(name)! / counts.get(name)!);
  return { names, mean_value };
}

export function loadData(dataName: string): Record<string, unknown> {
  const filePath = path.join(
    POSTERIORDB_PATH,
    "data",
    "data",
    `${dataName}.json.zip`,
  );
  return readZipJson(filePath) as Record<string, unknown>;
}

export function loadMeanStats(posteriorName: string): {
  names: string[];
  mean_value: number[];
} {
  const localStats = loadLocalMeanStats(posteriorName);
  if (localStats) {
    return localStats;
  }
  const filePath = path.join(
    POSTERIORDB_PATH,
    "reference_posteriors",
    "summary_statistics",
    "mean_value",
    "mean_value",
    `${posteriorName}.json`,
  );
  if (!fs.existsSync(filePath)) {
    throw new Error(
      `Missing posteriordb summary: ${filePath}. ` +
        "Set POSTERIORDB_PATH to a local posteriordb checkout or add a local " +
        `reference draw zip in ${LOCAL_REFERENCE_DRAWS}.`,
    );
  }
  const content = fs.readFileSync(filePath, "utf-8");
  return JSON.parse(content) as { names: string[]; mean_value: number[] };
}
