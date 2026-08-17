import { describe, expect, it, vi } from "vitest";

import { getModel } from "../../src/cache";
import type { ModelSpec } from "../../src/models";

describe("model cache", () => {
  it("reuses verified bytes across pages on the same origin", async () => {
    const bytes = new Uint8Array([1, 2, 3]);
    const digest = await crypto.subtle.digest("SHA-256", bytes);
    const sha256 = Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join("");
    const model: ModelSpec = {
      id: "test",
      file: "model.ort",
      bytes: bytes.byteLength,
      sha256,
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
