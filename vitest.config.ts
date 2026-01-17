import { defineConfig } from "vitest/config";
import { resolve } from "path";

export default defineConfig({
  resolve: {
    alias: {
      "jax-js-mcmc": resolve("/tmp/jax-js-mcmc/src/index.ts"),
      "@jax-js/jax": resolve("/tmp/jax-js/dist/index.js"),
    },
  },
  test: {
    testTimeout: 120000,
    hookTimeout: 30000,
    pool: "forks",
    poolOptions: {
      forks: {
        execArgv: ["--max-old-space-size=4096"],
      },
    },
  },
});
