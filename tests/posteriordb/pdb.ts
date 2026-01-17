import fs from "node:fs";
import path from "node:path";
import os from "node:os";
import AdmZip from "adm-zip";

const DEFAULT_PDB = path.join(os.homedir(), ".posteriordb", "posterior_database");

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
} | null {
  const filePath = path.join(
    POSTERIORDB_PATH,
    "reference_posteriors",
    "summary_statistics",
    "mean_value",
    "mean_value",
    `${posteriorName}.json`,
  );
  if (!fs.existsSync(filePath)) {
    return null;
  }
  const content = fs.readFileSync(filePath, "utf-8");
  return JSON.parse(content) as { names: string[]; mean_value: number[] };
}
