import { describe, expect, it } from "vitest";

import packageJson from "../../package.json";

import { models, selectModel } from "../../src/models";

describe("published production models", () => {
  it("selects explicit task-specific models", () => {
    expect(selectModel("classification", "int4")).toBe(models.classification.int4);
    expect(selectModel("classification", "int8")).toBe(models.classification.int8);
    expect(selectModel("regression", "int4")).toBe(models.regression.int4);
    expect(selectModel("regression", "int8")).toBe(models.regression.int8);
  });

  it("keeps repository-only FP32 out of the published catalog", () => {
    expect(selectModel("regression", "fp32").precision).toBe("fp32");
    expect("fp32" in models.classification).toBe(false);
    expect("fp32" in models.regression).toBe(false);
    expect(selectModel("regression", "int4")).toBe(models.regression.int4);
    expect(selectModel("regression", "int8")).toBe(models.regression.int8);
  });

  it("keeps FP32 repository-only in the npm allowlist", () => {
    expect(packageJson.files).toContain("src/models/*-int4-*.onnx");
    expect(packageJson.files).toContain("src/models/*-int8-*.onnx");
    expect(packageJson.files.some((entry) => entry.includes("fp32") || entry === "src/models/*.onnx")).toBe(false);
  });
});
