import { numpy as np, tree, Array, type JsTree } from "@jax-js/jax";

export type { JsTree };

export function mapTree<T, U, Tree extends JsTree<T>>(
  fn: (...args: T[]) => U,
  treeArg: Tree,
  ...rest: Tree[]
): JsTree<U> {
  return tree.map(fn, treeArg, ...rest) as JsTree<U>;
}

export function treeAdd<T extends JsTree<Array>>(a: T, b: T): T {
  return mapTree((x: Array, y: Array) => x.add(y), a, b) as T;
}

export function treeSub<T extends JsTree<Array>>(a: T, b: T): T {
  return mapTree((x: Array, y: Array) => x.sub(y), a, b) as T;
}

export function treeMul<T extends JsTree<Array>>(a: T, b: T): T {
  return mapTree((x: Array, y: Array) => x.mul(y), a, b) as T;
}

export function treeMulScalar<T extends JsTree<Array>>(a: T, scalar: number): T {
  return mapTree((x: Array) => x.mul(scalar), a) as T;
}

export function treeZerosLike<T extends JsTree<Array>>(a: T): T {
  return mapTree((x: Array) => np.zerosLike(x.ref), a) as T;
}

export function treeOnesLike<T extends JsTree<Array>>(a: T): T {
  return mapTree((x: Array) => np.onesLike(x.ref), a) as T;
}

export function treeSqrt<T extends JsTree<Array>>(a: T): T {
  return mapTree((x: Array) => np.sqrt(x), a) as T;
}

export function treeFlatten<T>(a: JsTree<T>): [T[], ReturnType<typeof tree.structure>] {
  return tree.flatten(a);
}

export function treeUnflatten<T>(
  treedef: ReturnType<typeof tree.structure>,
  leaves: T[],
): JsTree<T> {
  return tree.unflatten(treedef, leaves);
}

export function treeSum(a: JsTree<Array>): Array {
  const leaves = tree.leaves(a) as Array[];
  if (leaves.length === 0) {
    return np.array(0);
  }
  let acc = np.zeros([], { device: leaves[0].device });
  for (const leaf of leaves) {
    acc = acc.add(np.sum(leaf));
  }
  return acc;
}

export function treeDot(a: JsTree<Array>, b: JsTree<Array>): Array {
  const leavesA = tree.leaves(a) as Array[];
  const leavesB = tree.leaves(b) as Array[];
  if (leavesA.length !== leavesB.length) {
    throw new Error("treeDot: mismatched leaf counts");
  }
  if (leavesA.length === 0) {
    return np.array(0);
  }
  let acc = np.zeros([], { device: leavesA[0].device });
  for (let i = 0; i < leavesA.length; i++) {
    acc = acc.add(np.sum(leavesA[i].ref.mul(leavesB[i].ref)));
  }
  return acc;
}

export function treeRef<T extends JsTree<Array>>(a: T): T {
  return tree.ref(a) as T;
}

export function treeClone<T extends JsTree<Array>>(a: T): T {
  return mapTree((x: Array) => x.ref.add(0), a) as T;
}

export function treeDispose<T extends JsTree<Array>>(a: T): void {
  tree.dispose(a);
}

export function stackTrees<T extends JsTree<Array>>(
  trees: T[],
  axis: number = 0,
): T {
  if (trees.length === 0) {
    throw new Error("stackTrees: need at least one tree to stack");
  }
  const [baseLeaves, treedef] = tree.flatten(trees[0]);
  const allLeaves = trees.map((t) => tree.flatten(t)[0]);
  const stackedLeaves = baseLeaves.map((_leaf, i) =>
    np.stack(allLeaves.map((leaves) => leaves[i]), axis),
  );
  return tree.unflatten(treedef, stackedLeaves) as T;
}

export function prod(nums: number[]): number {
  return nums.reduce((a, b) => a * b, 1);
}

export function flattenToArray<T extends JsTree<Array>>(params: T): Array {
  const [leaves] = treeFlatten(params);
  const flatLeaves = leaves.map((leaf) => np.reshape(leaf, [-1]));
  return np.concatenate(flatLeaves, 0);
}

export function unflattenFromArray<T extends JsTree<Array>>(
  flat: Array,
  template: T,
): T {
  const [templateLeaves, treedef] = treeFlatten(template);
  const shapes = templateLeaves.map((leaf) => leaf.shape as number[]);
  const sizes = shapes.map((s) => prod(s));

  // Compute split indices
  const indices: number[] = [];
  let cumSum = 0;
  for (let i = 0; i < sizes.length - 1; i++) {
    cumSum += sizes[i];
    indices.push(cumSum);
  }

  // Use np.split with the indices
  const splits = np.split(flat, indices);

  // Reshape each split to the correct shape
  const newLeaves = splits.map((slice, i) => np.reshape(slice, shapes[i]));

  return treeUnflatten(treedef, newLeaves) as T;
}
