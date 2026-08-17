import { afterEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  classify: vi.fn(async () => ({
    backend: "wasm" as const,
    inferenceMs: 12,
    probabilities: [[0.2, 0.8]],
    predictions: [1],
  })),
  regress: vi.fn(async () => ({
    backend: "wasm" as const,
    inferenceMs: 15,
    predictions: [4.25],
  })),
  createSession: vi.fn(),
  dispose: vi.fn(async () => undefined),
}));

vi.mock("../../src/cache", () => ({
  getModel: vi.fn(async () => ({
    bytes: new Uint8Array([1, 2, 3]),
    fromCache: true,
    elapsedMs: 2,
    url: "https://example.test/package/model.onnx",
  })),
  isModelCached: vi.fn(async () => true),
  clearModelCache: vi.fn(async () => true),
}));

vi.mock("../../src/runtime", () => ({
  TabPFNSession: { create: mocks.createSession },
}));

import { load } from "../../src";

describe("task-aware estimators", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.clearAllMocks();
  });

  it("encodes feature categories and classification labels", async () => {
    mocks.createSession.mockResolvedValue({
      backend: "wasm",
      predictClassification: mocks.classify,
      predictRegression: mocks.regress,
      dispose: mocks.dispose,
    });
    vi.stubGlobal("location", new URL("https://example.test/app/"));

    const classifier = await load({ task: "classification", backend: "wasm", precision: "int8", baseUrl: "https://example.test/package/" });
    classifier.fit([[1, "red"], [2, "blue"]], ["no", "yes"]);
    const result = await classifier.infer([[3, "unseen"]]);

    expect(classifier.classes).toEqual(["no", "yes"]);
    expect(result.predictions).toEqual(["yes"]);
    expect(mocks.classify).toHaveBeenCalledWith({
      x: [[1, 0], [2, 1], [3, Number.NaN]],
      yTrain: [0, 1],
    });
  });

  it("runs regression with raw finite targets", async () => {
    mocks.createSession.mockResolvedValue({
      backend: "wasm",
      predictClassification: mocks.classify,
      predictRegression: mocks.regress,
      dispose: mocks.dispose,
    });
    vi.stubGlobal("location", new URL("https://example.test/app/"));

    const regressor = await load({ task: "regression", backend: "wasm", precision: "int8", baseUrl: "https://example.test/package/" });
    regressor.fit([[1], [2]], [3.5, 5.5]);

    expect(await regressor.predict([[3]])).toEqual([4.25]);
    expect(regressor.info.model.task).toBe("regression");
    expect(regressor.info.model.precision).toBe("int8");
    expect(mocks.regress).toHaveBeenCalledWith({ x: [[1], [2], [3]], yTrain: [3.5, 5.5] });
  });

  it("rejects invalid fit inputs", async () => {
    mocks.createSession.mockResolvedValue({
      backend: "wasm",
      predictClassification: mocks.classify,
      predictRegression: mocks.regress,
      dispose: mocks.dispose,
    });
    vi.stubGlobal("location", new URL("https://example.test/app/"));
    const classifier = await load({ task: "classification", backend: "wasm", precision: "int8", baseUrl: "https://example.test/package/" });
    const regressor = await load({ task: "regression", backend: "wasm", precision: "int8", baseUrl: "https://example.test/package/" });

    expect(() => classifier.fit([[1], [2]], ["only"])).toThrow(/lengths differ/u);
    expect(() => regressor.fit([[1], [2]], [1, Number.NaN])).toThrow(/must be finite/u);
  });

  it("does not hide WebGPU setup failures", async () => {
    mocks.createSession.mockRejectedValueOnce(new Error("adapter unavailable"));
    vi.stubGlobal("navigator", { gpu: {} });
    vi.stubGlobal("location", new URL("https://example.test/app/"));

    await expect(load({ task: "regression", backend: "webgpu", precision: "int4", baseUrl: "https://example.test/package/" })).rejects.toThrow(
      "adapter unavailable",
    );

    expect(mocks.createSession).toHaveBeenCalledTimes(1);
    expect(mocks.createSession.mock.calls[0]?.slice(1, 3)).toEqual(["webgpu", "regression"]);
  });

  it("keeps the previous fitted state when refitting fails", async () => {
    mocks.createSession.mockResolvedValue({
      backend: "wasm",
      predictClassification: mocks.classify,
      predictRegression: mocks.regress,
      dispose: mocks.dispose,
    });
    vi.stubGlobal("location", new URL("https://example.test/app/"));
    const regressor = await load({ task: "regression", backend: "wasm", precision: "int8", baseUrl: "https://example.test/package/" });
    regressor.fit([[1], [2]], [3.5, 5.5]);

    expect(() => regressor.fit([[10], [20]], [1, Number.NaN])).toThrow(/must be finite/u);
    await regressor.predict([[3]]);

    expect(mocks.regress).toHaveBeenLastCalledWith({ x: [[1], [2], [3]], yTrain: [3.5, 5.5] });
  });

  it("releases the inference session", async () => {
    mocks.createSession.mockResolvedValue({
      backend: "wasm",
      predictClassification: mocks.classify,
      predictRegression: mocks.regress,
      dispose: mocks.dispose,
    });
    vi.stubGlobal("location", new URL("https://example.test/app/"));
    const classifier = await load({ task: "classification", backend: "wasm", precision: "int8", baseUrl: "https://example.test/package/" });

    await classifier.dispose();

    expect(mocks.dispose).toHaveBeenCalledOnce();
  });
});
