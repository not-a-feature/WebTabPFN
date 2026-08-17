import { clearModelCache, getModel, isModelCached } from "./cache";
import type { Backend, ModelSpec, Precision, Task } from "./models";
import { hasWebGpu, models, selectModel } from "./models";
import type { ClassificationLabel, FeatureMatrix, FitOptions } from "./preprocessing";
import { encodeLabels, FeaturePreprocessor, validateRegressionTargets } from "./preprocessing";
import type { InferenceInfo, RegressionResult } from "./runtime";
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
  readonly task: Task;
  readonly backend: Backend;
  readonly precision: Precision;
  readonly baseUrl?: string | URL;
  readonly runtimeBaseUrl?: string | URL;
  readonly cache?: boolean;
}

export interface CacheOptions {
  readonly task: Task;
  readonly precision: Precision;
  readonly baseUrl?: string | URL;
}

export interface LoadInfo {
  readonly backend: Backend;
  readonly model: ModelSpec;
  readonly fromCache: boolean;
  readonly modelLoadMs: number;
  readonly sessionCreateMs: number;
  readonly modelUrl: string;
}

export interface ClassificationResult extends InferenceInfo {
  readonly probabilities: readonly (readonly number[])[];
  readonly predictions: readonly ClassificationLabel[];
}

export class TabPFNClassifier {
  readonly info: LoadInfo;
  readonly #session: TabPFNSession;
  #preprocessor: FeaturePreprocessor | undefined;
  #xTrain: number[][] | undefined;
  #yTrain: readonly number[] | undefined;
  #classes: readonly ClassificationLabel[] | undefined;

  constructor(session: TabPFNSession, info: LoadInfo) {
    this.#session = session;
    this.info = info;
  }

  get classes(): readonly ClassificationLabel[] {
    if (this.#classes === undefined) throw new Error("fit() must be called first");
    return this.#classes;
  }

  fit(xTrain: FeatureMatrix, yTrain: readonly ClassificationLabel[], options: FitOptions = {}): this {
    assertTrainingRows(xTrain, yTrain);
    const preprocessor = FeaturePreprocessor.fit(xTrain, options.categoricalFeatures);
    const encoded = encodeLabels(yTrain);
    const transformed = preprocessor.transform(xTrain, "xTrain");
    this.#preprocessor = preprocessor;
    this.#xTrain = transformed;
    this.#yTrain = encoded.values;
    this.#classes = encoded.classes;
    return this;
  }

  async infer(xTest: FeatureMatrix): Promise<ClassificationResult> {
    if (this.#preprocessor === undefined || this.#xTrain === undefined || this.#yTrain === undefined || this.#classes === undefined) {
      throw new Error("fit() must be called first");
    }
    const transformed = this.#preprocessor.transform(xTest, "xTest");
    const result = await this.#session.predictClassification({ x: [...this.#xTrain, ...transformed], yTrain: this.#yTrain });
    const predictions = result.predictions.map((index) => {
      const label = this.#classes![index];
      if (label === undefined) throw new Error(`model predicted unknown class ${index}`);
      return label;
    });
    return { ...result, predictions };
  }

  async predict(xTest: FeatureMatrix): Promise<readonly ClassificationLabel[]> {
    return (await this.infer(xTest)).predictions;
  }

  async predictProba(xTest: FeatureMatrix): Promise<readonly (readonly number[])[]> {
    return (await this.infer(xTest)).probabilities;
  }

  async dispose(): Promise<void> {
    await this.#session.dispose();
  }
}

export class TabPFNRegressor {
  readonly info: LoadInfo;
  readonly #session: TabPFNSession;
  #preprocessor: FeaturePreprocessor | undefined;
  #xTrain: number[][] | undefined;
  #yTrain: number[] | undefined;

  constructor(session: TabPFNSession, info: LoadInfo) {
    this.#session = session;
    this.info = info;
  }

  fit(xTrain: FeatureMatrix, yTrain: readonly number[], options: FitOptions = {}): this {
    assertTrainingRows(xTrain, yTrain);
    const preprocessor = FeaturePreprocessor.fit(xTrain, options.categoricalFeatures);
    const transformed = preprocessor.transform(xTrain, "xTrain");
    const targets = validateRegressionTargets(yTrain);
    this.#preprocessor = preprocessor;
    this.#xTrain = transformed;
    this.#yTrain = targets;
    return this;
  }

  async infer(xTest: FeatureMatrix): Promise<RegressionResult> {
    if (this.#preprocessor === undefined || this.#xTrain === undefined || this.#yTrain === undefined) {
      throw new Error("fit() must be called first");
    }
    const transformed = this.#preprocessor.transform(xTest, "xTest");
    return this.#session.predictRegression({ x: [...this.#xTrain, ...transformed], yTrain: this.#yTrain });
  }

  async predict(xTest: FeatureMatrix): Promise<readonly number[]> {
    return (await this.infer(xTest)).predictions;
  }

  async dispose(): Promise<void> {
    await this.#session.dispose();
  }
}

export async function preload(options: LoadOptions): Promise<Omit<LoadInfo, "sessionCreateMs">> {
  const backend = options.backend;
  const model = selectModel(options.task, options.precision);
  const baseUrl = resolveUrl(options.baseUrl ?? defaultBaseUrl);
  const loaded = await getModel(model, baseUrl, options.cache !== false);
  return { backend, model, fromCache: loaded.fromCache, modelLoadMs: loaded.elapsedMs, modelUrl: loaded.url };
}

export function load(options: LoadOptions & { readonly task: "classification" }): Promise<TabPFNClassifier>;
export function load(options: LoadOptions & { readonly task: "regression" }): Promise<TabPFNRegressor>;
export async function load(options: LoadOptions): Promise<TabPFNClassifier | TabPFNRegressor> {
  return loadWithBackend(options, options.backend);
}

async function loadWithBackend(
  options: LoadOptions,
  backend: Backend,
): Promise<TabPFNClassifier | TabPFNRegressor> {
  const model = selectModel(options.task, options.precision);
  const baseUrl = resolveUrl(options.baseUrl ?? defaultBaseUrl);
  const loaded = await getModel(model, baseUrl, options.cache !== false);
  const sessionStarted = performance.now();
  const session = await TabPFNSession.create(
    loaded.bytes,
    backend,
    model.task,
    resolveUrl(options.runtimeBaseUrl ?? baseUrl),
  );
  const info: LoadInfo = {
    backend,
    model,
    fromCache: loaded.fromCache,
    modelLoadMs: loaded.elapsedMs,
    sessionCreateMs: performance.now() - sessionStarted,
    modelUrl: loaded.url,
  };
  return options.task === "classification" ? new TabPFNClassifier(session, info) : new TabPFNRegressor(session, info);
}

export async function isCached(options: CacheOptions): Promise<boolean> {
  const model = selectModel(options.task, options.precision);
  return isModelCached(model, resolveUrl(options.baseUrl ?? defaultBaseUrl));
}

export const clearCache = clearModelCache;
export { hasWebGpu, models };
export type { Backend, ModelSpec, Precision, Task } from "./models";
export type { ClassificationLabel, FeatureMatrix, FeatureValue, FitOptions } from "./preprocessing";
export type { RegressionResult } from "./runtime";

function assertTrainingRows(xTrain: FeatureMatrix, yTrain: readonly unknown[]): void {
  if (xTrain.length !== yTrain.length) throw new Error("xTrain and yTrain lengths differ");
  if (xTrain.length < 2) throw new Error("training data must contain at least two rows");
}

function resolveUrl(value: string | URL): string {
  return new URL(value, pageUrl).href;
}
