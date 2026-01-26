#!/usr/bin/env bash
set -euo pipefail

REPO="/tmp/jax-js-mcmc-2"
if [ ! -d "$REPO/.git" ]; then
  echo "Cloning jax-js-mcmc-2 into $REPO..."
  git clone https://github.com/StefanSko/jax-js-mcmc-2.git "$REPO"
fi

cd "$REPO"

JAXJS_CACHE_LOG=1 NODE_OPTIONS="--expose-gc --loader ./tools/jaxjs-loader.mjs" \
  ITERATIONS=2000 LOG_EVERY=500 npx tsx examples/memory-profile-hmc-jit-step.ts
