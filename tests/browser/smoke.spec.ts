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
  const first = await page.evaluate(() => window.WebTabPFN.preload({ backend: "wasm", precision: "int8" }));
  expect(first.fromCache).toBe(false);

  const secondPage = await page.context().newPage();
  await secondPage.goto("/benchmark/?second-tab");
  expect(await secondPage.evaluate(() => window.WebTabPFN.isCached("int8"))).toBe(true);
  await secondPage.close();
});

test("the standalone benchmark runs the shipped unquantized WASM model", async ({ page }) => {
  await page.goto("/benchmark/");
  const result = await page.evaluate(() => window.runWebTabPFNBenchmark({
    caseLimit: 1,
    configurations: [{ backend: "wasm", precision: "fp32" }],
    runs: 1,
    warmups: 0,
  }));

  expect(result.configurations[0]?.status).toBe("ok");
  expect(result.configurations[0]?.cases?.[0]?.timingMs.count).toBe(1);
  expect(result.configurations[0]?.cases?.[0]?.metrics.accuracy).toBeGreaterThan(0);
});
