import { expect, test } from "@playwright/test";

test("benchmark page exposes the browser runner", async ({ page }) => {
  await page.goto("/benchmark/");
  await expect(page.locator("#status")).toHaveText("Benchmark runtime ready");
  expect(await page.evaluate(() => typeof window.runWebTabPFNBenchmark)).toBe("function");
  expect(await page.evaluate(() => typeof window.WebTabPFN.load)).toBe("function");
});

test("a model cached in one tab is available to another tab on the origin", async ({ page }) => {
  await page.goto("/benchmark/");
  await page.evaluate(() => window.WebTabPFN.clearCache());
  const first = await page.evaluate(() => window.WebTabPFN.preload({ task: "classification", backend: "wasm", precision: "int8" }));
  expect(first.fromCache).toBe(false);

  const secondPage = await page.context().newPage();
  await secondPage.goto("/benchmark/?second-tab");
  expect(await secondPage.evaluate(() => window.WebTabPFN.isCached({ task: "classification", precision: "int8" }))).toBe(true);
  await secondPage.close();
});

test("the standalone benchmark runs the repository FP32 WASM model", async ({ page }) => {
  await page.goto("/benchmark/");
  const result = await page.evaluate(() => window.runWebTabPFNBenchmark({
    caseLimit: 1,
    configurations: [{ task: "classification", backend: "wasm", precision: "fp32" }],
    runs: 1,
    warmups: 0,
  }));

  expect(result.configurations[0]?.status).toBe("ok");
  expect(result.configurations[0]?.cases?.[0]?.timingMs.count).toBe(1);
  expect(result.configurations[0]?.cases?.[0]?.metrics.accuracy).toBeGreaterThan(0);
});

test("published classification models preserve FP32 WASM quality", async ({ page }) => {
  await page.goto("/benchmark/");
  const result = await page.evaluate(() => window.runWebTabPFNBenchmark({
    configurations: ["fp32", "int4", "int8"].map((precision) => ({ task: "classification", backend: "wasm", precision })),
    runs: 1,
    warmups: 0,
  }));

  const fp32 = result.configurations.find((configuration) => configuration.precision === "fp32");
  expect(fp32?.status).toBe("ok");
  const fp32Accuracy = mean((fp32?.cases ?? []).map(({ metrics }) => metrics.accuracy ?? 0));
  for (const configuration of result.configurations) {
    expect(configuration.status).toBe("ok");
    expect(mean((configuration.cases ?? []).map(({ metrics }) => metrics.accuracy ?? 0))).toBeGreaterThanOrEqual(
      fp32Accuracy - 0.02,
    );
    for (const benchmarkCase of configuration.cases ?? []) {
      expect(Number.isFinite(benchmarkCase.parity?.probabilityMae)).toBe(true);
    }
  }
});

test("all repository WASM regressors preserve useful raw-space quality", async ({ page }) => {
  await page.goto("/benchmark/");
  const result = await page.evaluate(() => window.runWebTabPFNBenchmark({
    referenceUrl: "./regression/reference.json",
    configurations: ["fp32", "int4", "int8"].map((precision) => ({ task: "regression", backend: "wasm", precision })),
    runs: 1,
    warmups: 0,
  }));

  const fp32 = result.configurations.find((configuration) => configuration.precision === "fp32");
  expect(fp32?.status).toBe("ok");
  for (const configuration of result.configurations) {
    expect(configuration.status).toBe("ok");
    for (const [index, benchmarkCase] of (configuration.cases ?? []).entries()) {
      expect(benchmarkCase.timingMs.count).toBe(1);
      expect(Number.isFinite(benchmarkCase.metrics.r2)).toBe(true);
      expect(Number.isFinite(benchmarkCase.parity?.predictionMae)).toBe(true);
      expect(benchmarkCase.metrics.r2).toBeGreaterThanOrEqual((fp32?.cases?.[index]?.metrics.r2 ?? 0) - 0.06);
    }
  }
});

test("all repository regressors run on WebGPU when the browser exposes it", async ({ page }) => {
  await page.goto("/benchmark/");
  const available = await page.evaluate(async () => {
    const candidate = navigator as Navigator & { gpu?: { requestAdapter: () => Promise<unknown | null> } };
    return candidate.gpu !== undefined && await candidate.gpu.requestAdapter() !== null;
  });
  test.skip(!available, "WebGPU has no usable adapter in this browser");
  const predictions = await page.evaluate(async () => {
    const results = [];
    for (const precision of ["fp32", "int4", "int8"] as const) {
      const regressor = await window.WebTabPFN.load({ task: "regression", backend: "webgpu", precision }) as {
        fit: (x: number[][], y: number[]) => void;
        predict: (x: number[][]) => Promise<number[]>;
      };
      regressor.fit([[0], [1], [2], [3]], [1, 3, 5, 7]);
      results.push(await regressor.predict([[1.5]]));
    }
    return results;
  });
  expect(predictions).toHaveLength(3);
  expect(predictions.every((values) => values.length === 1 && Number.isFinite(values[0]))).toBe(true);
});

function mean(values: number[]): number {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}
