import { numpy as np } from "@jax-js/jax";
import { model, observed, param } from "../../src/model";
import { normal } from "../../src/distributions";

const simple = model({
  mu: param(normal(0, 1)),
  y: observed(({ mu }) => normal(mu, 1)),
});

const predictive = simple.bind({});

// @ts-expect-error logProb should not exist on predictive models
predictive.logProb({ mu: np.array(0) });
