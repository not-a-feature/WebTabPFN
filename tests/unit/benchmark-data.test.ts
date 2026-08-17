import { readFileSync } from "node:fs";

import { describe, expect, it } from "vitest";

const reference = JSON.parse(
  readFileSync(new URL("../../benchmark/classification/reference.json", import.meta.url), "utf8"),
) as {
  schemaVersion: number;
  task: string;
  cases: Array<{
    name: string;
    x: number[][];
    yTest: number[];
    yTrain: number[];
  }>;
};

const regressionReference = JSON.parse(
  readFileSync(new URL("../../benchmark/regression/reference.json", import.meta.url), "utf8"),
) as {
  task: string;
  cases: Array<{
    name: string;
    x: number[][];
    yTest: number[];
    yTrain: number[];
    python: { predictions: number[] };
  }>;
};

const regressionResults = JSON.parse(
  readFileSync(new URL("../../benchmark/regression/results.json", import.meta.url), "utf8"),
) as {
  task: string;
  native: Record<string, {
    referenceLabel?: string;
    checkpointSha256: string | null;
    environment: { device: string; hostname: string; gpu?: { name: string } };
    benchmark: { runs: number; warmups: number };
    cases: Array<{ name: string; python: { metrics: { r2: number } } }>;
  }>;
  browser: {
    laptop: {
      configurations: Array<{
        backend: string;
        precision: string;
        status: string;
        cases: Array<{ metrics: { r2: number } }>;
      }>;
    };
  };
};

describe("benchmark reference datasets", () => {
  it("contains eight deterministic classification cases", () => {
    expect(reference.schemaVersion).toBe(2);
    expect(reference.task).toBe("classification");
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

  it("labels every classification GPU baseline as Native H100 GPU", () => {
    for (const name of ["latency-percentiles", "accuracy", "mcc", "pareto-speed-vs-accuracy"]) {
      const svg = readFileSync(
        new URL(`../../benchmark/classification/plots/${name}.svg`, import.meta.url),
        "utf8",
      );
      expect(svg).toContain("Native H100 GPU");
    }
  });

  it("keeps the laptop CPU out of classification quality plots only", () => {
    const plot = (name: string): string => readFileSync(
      new URL(`../../benchmark/classification/plots/${name}.svg`, import.meta.url),
      "utf8",
    );
    expect(plot("accuracy")).not.toContain("native laptop-cpu");
    expect(plot("mcc")).not.toContain("native laptop-cpu");
    expect(plot("latency-percentiles")).toContain("native laptop-cpu");
    expect(plot("pareto-speed-vs-accuracy")).toContain("native laptop-cpu");
  });
});

describe("regression benchmark reference", () => {
  it("contains deterministic finite Python reference predictions", () => {
    expect(regressionReference.task).toBe("regression");
    expect(regressionReference.cases.map(({ name }) => name)).toEqual([
      "diabetes",
      "synthetic",
      "synthetic_offset",
    ]);
    for (const benchmarkCase of regressionReference.cases) {
      expect(benchmarkCase.x.length).toBe(benchmarkCase.yTrain.length + benchmarkCase.yTest.length);
      expect(benchmarkCase.python.predictions).toHaveLength(benchmarkCase.yTest.length);
      expect(benchmarkCase.x.flat().every(Number.isFinite)).toBe(true);
      expect(benchmarkCase.yTrain.every(Number.isFinite)).toBe(true);
      expect(benchmarkCase.python.predictions.every(Number.isFinite)).toBe(true);
    }
  });

  it("persists the complete passing browser precision matrix", () => {
    const configurations = regressionResults.browser.laptop.configurations;
    expect(regressionResults.task).toBe("regression");
    expect(configurations).toHaveLength(6);
    expect(new Set(configurations.map(({ backend }) => backend))).toEqual(new Set(["wasm", "webgpu"]));
    expect(new Set(configurations.map(({ precision }) => precision))).toEqual(new Set(["fp32", "int4", "int8"]));
    for (const configuration of configurations) {
      expect(configuration.status).toBe("ok");
      expect(configuration.cases).toHaveLength(3);
      expect(configuration.cases.every(({ metrics }) => Number.isFinite(metrics.r2))).toBe(true);
    }
  });

  it("uses the pinned native H100 GPU run as the reference", () => {
    const referenceRuns = Object.values(regressionResults.native).filter(
      ({ referenceLabel }) => referenceLabel !== undefined,
    );
    expect(referenceRuns).toHaveLength(1);
    const [nativeH100] = referenceRuns;
    expect(nativeH100?.referenceLabel).toBe("Native H100 GPU");
    expect(nativeH100?.checkpointSha256).toBe(
      "2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736",
    );
    expect(nativeH100?.environment.hostname).toBe("redacted");
    expect(nativeH100?.environment.device).toBe("cuda:0");
    expect(nativeH100?.environment.gpu?.name).toBe("NVIDIA H100 PCIe");
    expect(nativeH100?.benchmark).toEqual({ runs: 5, warmups: 1 });
    expect(nativeH100?.cases.map(({ name }) => name)).toEqual([
      "diabetes",
      "synthetic",
      "synthetic_offset",
    ]);
    expect(nativeH100?.cases.every(({ python }) => Number.isFinite(python.metrics.r2))).toBe(true);
  });

  it("ships regression plots with dashed native GPU lines", () => {
    for (const name of ["latency-percentiles", "r2", "errors", "pareto-speed-vs-r2"]) {
      const svg = readFileSync(
        new URL(`../../benchmark/regression/plots/${name}.svg`, import.meta.url),
        "utf8",
      );
      expect(svg).toContain("<svg");
    }
    const plot = (name: string): string => readFileSync(
      new URL(`../../benchmark/regression/plots/${name}.svg`, import.meta.url),
      "utf8",
    );
    const latency = plot("latency-percentiles");
    const r2 = plot("r2");
    const errors = plot("errors");
    expect(latency).toContain("Native H100 GPU p50");
    expect(latency).toContain("browser webgpu/int4");
    expect(latency).toContain('class="bar-value"');
    expect(latency).toContain('stroke-dasharray="8 6"');
    expect(r2).toContain("Native H100 GPU R²");
    expect(r2).toContain('stroke-dasharray="8 6"');
    expect(errors).toContain("Native H100 GPU MAE");
    expect(errors).toContain("Native H100 GPU RMSE");
    expect(errors).toContain('stroke-dasharray="8 6"');
  });
});
