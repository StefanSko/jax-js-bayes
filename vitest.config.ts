import path from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vitest/config";

const rootDir = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  resolve: {
    alias: {
      "jax-js-mcmc": path.join(rootDir, "src/mcmc/index.ts"),
      "jax-js-mcmc-2": path.join(rootDir, "node_modules/jax-js-mcmc-2/src/index.ts"),
    },
  },
  test: {
    pool: "forks",
    poolOptions: {
      forks: {
        singleFork: true,
      },
    },
  },
});
