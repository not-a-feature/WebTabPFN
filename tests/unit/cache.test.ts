import { describe, expect, it, vi } from "vitest";

import { getModel } from "../../src/cache";
import type { ModelSpec } from "../../src/models";

describe("model cache", () => {
  it("reuses model bytes across pages on the same origin", async () => {
    const bytes = new Uint8Array([1, 2, 3]);
    const model: ModelSpec = {
      id: "test",
      task: "classification",
      file: "model.ort",
      bytes: bytes.byteLength,
      precision: "int8",
    };
    vi.stubGlobal("location", new URL("https://example.test/benchmark/"));
    const match = vi.fn().mockResolvedValue(new Response(bytes));
    const put = vi.fn().mockResolvedValue(undefined);
    vi.stubGlobal("caches", { open: vi.fn().mockResolvedValue({ match, put }) });

    const loaded = await getModel(model, "https://example.test/package/", true);

    expect(match).toHaveBeenCalledWith("https://example.test/package/model.ort");
    expect(loaded.fromCache).toBe(true);
    vi.unstubAllGlobals();
  });
});
