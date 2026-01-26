#!/usr/bin/env bash
set -euo pipefail

NODE_OPTIONS_BASE=${NODE_OPTIONS:-}
export NODE_OPTIONS="${NODE_OPTIONS_BASE} --max-old-space-size=8192"
export VITEST_MAX_THREADS=${VITEST_MAX_THREADS:-1}
export VITEST_MIN_THREADS=${VITEST_MIN_THREADS:-1}

pnpm vitest run tests/posteriordb/eight-schools.smoke.test.ts
pnpm vitest run tests/posteriordb/posteriordb.test.ts
