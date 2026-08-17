import { afterEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  predict: vi.fn(async () => ({
    backend: "wasm" as const,
    inferenceMs: 12,
    probabilities: [[0.2, 0.8]],
    predictions: [1],
  })),
  createSession: vi.fn(),
}));

vi.mock("../../src/cache", () => ({
  getModel: vi.fn(async () => ({
    bytes: new Uint8Array([1, 2, 3]),
    fromCache: true,
    elapsedMs: 2,
    url: "https://example.test/package/tabpfn-v2-classifier-int8.onnx",
  })),
  isModelCached: vi.fn(async () => true),
  clearModelCache: vi.fn(async () => true),
}));

vi.mock("../../src/runtime", () => ({
  softmax: vi.fn(),
  TabPFNSession: {
    create: mocks.createSession,
  },
}));

import { load } from "../../src";

describe("TabPFNClassifier", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.clearAllMocks();
  });

  it("owns model setup and exposes a fit/infer consumer API", async () => {
    mocks.createSession.mockResolvedValue({
      backend: "wasm",
      fallbackReason: undefined,
      predict: mocks.predict,
    });
    vi.stubGlobal("location", new URL("https://example.test/app/"));

    const classifier = await load({ backend: "wasm", baseUrl: "https://example.test/package/" });
    classifier.fit([[1, 2], [3, 4]], [0, 1]);
    const result = await classifier.infer([[5, 6]]);

    expect(classifier.info.model.precision).toBe("int8");
    expect(result.predictions).toEqual([1]);
    expect(mocks.predict).toHaveBeenCalledWith({ x: [[1, 2], [3, 4], [5, 6]], yTrain: [0, 1] });
  });

  it("rejects mismatched training rows and labels", async () => {
    mocks.createSession.mockResolvedValue({
      backend: "wasm",
      fallbackReason: undefined,
      predict: mocks.predict,
    });
    vi.stubGlobal("location", new URL("https://example.test/app/"));
    const classifier = await load({ backend: "wasm", baseUrl: "https://example.test/package/" });

    expect(() => classifier.fit([[1], [2]], [0])).toThrow(/lengths differ/u);
  });

  it("falls back to WASM when automatic WebGPU setup fails", async () => {
    mocks.createSession
      .mockRejectedValueOnce(new Error("adapter unavailable"))
      .mockResolvedValueOnce({ backend: "wasm", fallbackReason: "WebGPU failed", predict: mocks.predict });
    vi.stubGlobal("navigator", { gpu: {} });
    vi.stubGlobal("location", new URL("https://example.test/app/"));

    const classifier = await load({ baseUrl: "https://example.test/package/" });

    expect(mocks.createSession).toHaveBeenCalledTimes(2);
    expect(mocks.createSession.mock.calls[0]?.[1]).toBe("webgpu");
    expect(mocks.createSession.mock.calls[1]?.[1]).toBe("wasm");
    expect(classifier.info.backend).toBe("wasm");
    expect(classifier.info.fallbackReason).toContain("adapter unavailable");
  });
});
