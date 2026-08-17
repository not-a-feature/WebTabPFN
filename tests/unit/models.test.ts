import { describe, expect, it } from "vitest";

import { models, selectModel } from "../../src/models";

describe("embedded production models", () => {
  it("selects the compact WebGPU default", () => {
    expect(selectModel("webgpu")).toBe(models.int4);
  });

  it("allows every backend/model combination for benchmarking", () => {
    expect(selectModel("wasm", "int4")).toBe(models.int4);
    expect(selectModel("webgpu", "int8")).toBe(models.int8);
    expect(selectModel("webgpu", "fp32")).toBe(models.fp32);
    expect(selectModel("wasm", "fp32")).toBe(models.fp32);
  });
});
