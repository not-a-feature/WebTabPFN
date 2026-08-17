import { readFileSync } from "node:fs";

import { describe, expect, it } from "vitest";

const reference = JSON.parse(
  readFileSync(new URL("../../benchmark/reference.json", import.meta.url), "utf8"),
) as {
  cases: Array<{
    name: string;
    x: number[][];
    yTest: number[];
    yTrain: number[];
  }>;
};

describe("benchmark reference datasets", () => {
  it("contains eight deterministic classification cases", () => {
    expect(reference.cases.map(({ name }) => name)).toEqual([
      "breast_cancer",
      "wine",
      "synthetic_4class",
      "iris",
      "digits",
      "moons",
      "circles",
      "synthetic_imbalanced_3class",
    ]);
  });

  it("contains finite matrices and dense labels for every case", () => {
    for (const benchmarkCase of reference.cases) {
      expect(benchmarkCase.x.length).toBe(
        benchmarkCase.yTrain.length + benchmarkCase.yTest.length,
      );
      expect(benchmarkCase.yTest).toHaveLength(48);
      expect(benchmarkCase.x.every((row) => row.length === benchmarkCase.x[0]?.length)).toBe(true);
      expect(benchmarkCase.x.flat().every(Number.isFinite)).toBe(true);
      const labels = [...new Set(benchmarkCase.yTrain)].sort((left, right) => left - right);
      expect(labels).toEqual(Array.from({ length: labels.length }, (_, index) => index));
    }
  });
});
