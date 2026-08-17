import { clearModelCache, getModel, isModelCached } from "./cache";
import type { Backend, ConcreteBackend, ModelSpec, Precision } from "./models";
import { hasWebGpu, models, resolveBackend, selectModel } from "./models";
import type { PredictionResult } from "./runtime";
import { TabPFNSession } from "./runtime";

const currentScript = typeof document === "undefined" ? undefined : document.currentScript;
const pageUrl = typeof location === "undefined" ? "http://localhost/" : location.href;
const scriptUrl = typeof HTMLScriptElement !== "undefined"
  && currentScript instanceof HTMLScriptElement
  && currentScript.src !== ""
  ? currentScript.src
  : pageUrl;
const defaultBaseUrl = new URL(".", scriptUrl).href;

export interface LoadOptions {
  readonly backend?: Backend;
  readonly precision?: Precision;
  readonly baseUrl?: string | URL;
  readonly runtimeBaseUrl?: string | URL;
  readonly cache?: boolean;
}

export interface LoadInfo {
  readonly backend: ConcreteBackend;
  readonly fallbackReason?: string;
  readonly model: ModelSpec;
  readonly fromCache: boolean;
  readonly modelLoadMs: number;
  readonly sessionCreateMs: number;
  readonly modelUrl: string;
}

export class TabPFNClassifier {
  readonly info: LoadInfo;
  readonly #session: TabPFNSession;
  #xTrain: number[][] | undefined;
  #yTrain: number[] | undefined;

  constructor(session: TabPFNSession, info: LoadInfo) {
    this.#session = session;
    this.info = info;
  }

  fit(xTrain: readonly (readonly number[])[], yTrain: readonly number[]): this {
    if (xTrain.length !== yTrain.length) throw new Error("xTrain and yTrain lengths differ");
    if (xTrain.length < 2) throw new Error("training data must contain at least two rows");
    this.#xTrain = xTrain.map((row) => [...row]);
    this.#yTrain = [...yTrain];
    return this;
  }

  async infer(xTest: readonly (readonly number[])[]): Promise<PredictionResult> {
    if (this.#xTrain === undefined || this.#yTrain === undefined) throw new Error("fit() must be called first");
    if (xTest.length === 0) throw new Error("xTest must contain at least one row");
    return this.#session.predict({ x: [...this.#xTrain, ...xTest], yTrain: this.#yTrain });
  }

  async predict(xTest: readonly (readonly number[])[]): Promise<readonly number[]> {
    return (await this.infer(xTest)).predictions;
  }

  async predictProba(xTest: readonly (readonly number[])[]): Promise<readonly (readonly number[])[]> {
    return (await this.infer(xTest)).probabilities;
  }
}

export async function preload(options: LoadOptions = {}): Promise<Omit<LoadInfo, "sessionCreateMs">> {
  const backend = resolveBackend(options.backend);
  const model = selectModel(backend, options.precision);
  const baseUrl = resolveUrl(options.baseUrl ?? defaultBaseUrl);
  const loaded = await getModel(model, baseUrl, options.cache !== false);
  return {
    backend,
    model,
    fromCache: loaded.fromCache,
    modelLoadMs: loaded.elapsedMs,
    modelUrl: loaded.url,
  };
}

export async function load(options: LoadOptions = {}): Promise<TabPFNClassifier> {
  const requested = options.backend ?? "auto";
  const backend = resolveBackend(requested);
  if (requested !== "auto" || backend === "wasm") return loadWithBackend(options, backend);
  try {
    return await loadWithBackend(options, "webgpu");
  } catch (error) {
    const reason = error instanceof Error ? error.message : String(error);
    return loadWithBackend(options, "wasm", `WebGPU failed: ${reason}`);
  }
}

async function loadWithBackend(
  options: LoadOptions,
  backend: ConcreteBackend,
  fallbackReason?: string,
): Promise<TabPFNClassifier> {
  const model = selectModel(backend, options.precision);
  const baseUrl = resolveUrl(options.baseUrl ?? defaultBaseUrl);
  const loaded = await getModel(model, baseUrl, options.cache !== false);
  const sessionStarted = performance.now();
  const session = await TabPFNSession.create(
    loaded.bytes,
    backend,
    resolveUrl(options.runtimeBaseUrl ?? baseUrl),
    fallbackReason,
  );
  return new TabPFNClassifier(session, {
    backend,
    ...(fallbackReason === undefined ? {} : { fallbackReason }),
    model,
    fromCache: loaded.fromCache,
    modelLoadMs: loaded.elapsedMs,
    sessionCreateMs: performance.now() - sessionStarted,
    modelUrl: loaded.url,
  });
}

export async function isCached(precision?: Precision, baseUrl: string | URL = defaultBaseUrl): Promise<boolean> {
  const model = models[precision ?? (hasWebGpu() ? "int4" : "int8")];
  return isModelCached(model, resolveUrl(baseUrl));
}

export const clearCache = clearModelCache;
export { hasWebGpu, models };
export type { Backend, ConcreteBackend, ModelSpec, Precision, PredictionResult };

function resolveUrl(value: string | URL): string {
  return new URL(value, pageUrl).href;
}
