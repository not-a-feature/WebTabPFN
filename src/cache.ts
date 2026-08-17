import type { ModelSpec } from "./models";

const CACHE_NAME = "webtabpfn-v2-f65a3568";

export interface CachedModel {
  readonly bytes: Uint8Array;
  readonly fromCache: boolean;
  readonly elapsedMs: number;
  readonly url: string;
}

export async function getModel(spec: ModelSpec, baseUrl: string, useCache: boolean): Promise<CachedModel> {
  const url = new URL(spec.file, baseUrl).href;
  const started = performance.now();
  const cache = useCache ? await caches.open(CACHE_NAME) : undefined;
  const cached = await cache?.match(url);
  if (cached !== undefined) {
    const bytes = new Uint8Array(await cached.arrayBuffer());
    assertLength(bytes, spec);
    return { bytes, fromCache: true, elapsedMs: performance.now() - started, url };
  }

  const response = await fetch(url);
  if (!response.ok) throw new Error(`model download failed: ${response.status} ${response.statusText}`);
  const bytes = new Uint8Array(await response.arrayBuffer());
  assertLength(bytes, spec);
  const digest = await sha256(bytes);
  if (digest !== spec.sha256) throw new Error(`model SHA-256 mismatch: expected ${spec.sha256}, received ${digest}`);
  await cache?.put(url, new Response(bytes, { headers: { "Content-Type": "application/octet-stream" } }));
  return { bytes, fromCache: false, elapsedMs: performance.now() - started, url };
}

export async function isModelCached(spec: ModelSpec, baseUrl: string): Promise<boolean> {
  return (await (await caches.open(CACHE_NAME)).match(new URL(spec.file, baseUrl).href)) !== undefined;
}

export async function clearModelCache(): Promise<boolean> {
  return caches.delete(CACHE_NAME);
}

function assertLength(bytes: Uint8Array, spec: ModelSpec): void {
  if (bytes.byteLength !== spec.bytes) {
    throw new Error(`model length mismatch: expected ${spec.bytes}, received ${bytes.byteLength}`);
  }
}

async function sha256(bytes: Uint8Array): Promise<string> {
  const copy = new Uint8Array(bytes);
  const digest = await crypto.subtle.digest("SHA-256", copy.buffer);
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join("");
}
