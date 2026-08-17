import * as ort from "onnxruntime-web/webgpu";

import type { ConcreteBackend } from "./models";

export interface TabularBatch {
  readonly x: readonly (readonly number[])[];
  readonly yTrain: readonly number[];
}

export interface PredictionResult {
  readonly backend: ConcreteBackend;
  readonly fallbackReason?: string;
  readonly inferenceMs: number;
  readonly probabilities: readonly (readonly number[])[];
  readonly predictions: readonly number[];
}

interface ValidatedBatch {
  readonly x: Float32Array;
  readonly y: Float32Array;
  readonly totalRows: number;
  readonly trainRows: number;
  readonly features: number;
  readonly classes: number;
}

export class TabPFNSession {
  readonly backend: ConcreteBackend;
  readonly fallbackReason: string | undefined;
  readonly #session: ort.InferenceSession;

  private constructor(session: ort.InferenceSession, backend: ConcreteBackend, fallbackReason?: string) {
    this.#session = session;
    this.backend = backend;
    this.fallbackReason = fallbackReason;
  }

  static async create(
    model: Uint8Array,
    backend: ConcreteBackend,
    runtimeBaseUrl: string,
    fallbackReason?: string,
  ): Promise<TabPFNSession> {
    ort.env.wasm.numThreads = 0;
    ort.env.wasm.wasmPaths = {
      mjs: new URL("ort-wasm-simd-threaded.asyncify.mjs", runtimeBaseUrl).href,
      wasm: new URL("ort-wasm-simd-threaded.asyncify.wasm", runtimeBaseUrl).href,
    };
    if (backend === "wasm") {
      return new TabPFNSession(await createSession(model, ["wasm"]), "wasm", fallbackReason);
    }
    if (!("gpu" in navigator)) throw new Error("WebGPU was requested but navigator.gpu is unavailable");
    return new TabPFNSession(await createSession(model, ["webgpu"]), "webgpu");
  }

  async predict(batch: TabularBatch): Promise<PredictionResult> {
    const input = validateBatch(batch);
    const feeds: Record<string, ort.Tensor> = {
      x: new ort.Tensor("float32", input.x, [1, input.totalRows, input.features]),
      y: new ort.Tensor("float32", input.y, [1, input.trainRows]),
    };
    const started = performance.now();
    const outputs = await this.#session.run(feeds);
    const inferenceMs = performance.now() - started;
    const logits = outputs.logits;
    if (logits === undefined) throw new Error("model output 'logits' is missing");
    if (logits.dims.length !== 3 || logits.dims[0] !== 1 || logits.dims[1] !== input.totalRows) {
      throw new Error(`unexpected logits shape: ${logits.dims.join("x")}`);
    }
    if (!(logits.data instanceof Float32Array)) throw new Error(`expected float32 logits, received ${logits.type}`);
    const modelClasses = logits.dims[2]!;
    if (modelClasses < input.classes) throw new Error("model output has fewer classes than the training labels");

    const probabilities: number[][] = [];
    const predictions: number[] = [];
    for (let row = input.trainRows; row < input.totalRows; row += 1) {
      const offset = row * modelClasses;
      const values = Array.from(logits.data.slice(offset, offset + input.classes));
      const probability = softmax(values);
      probabilities.push(probability);
      predictions.push(argmax(probability));
    }
    return {
      backend: this.backend,
      ...(this.fallbackReason === undefined ? {} : { fallbackReason: this.fallbackReason }),
      inferenceMs,
      probabilities,
      predictions,
    };
  }
}

async function createSession(model: Uint8Array, executionProviders: readonly ConcreteBackend[]): Promise<ort.InferenceSession> {
  return ort.InferenceSession.create(model, {
    executionProviders: [...executionProviders],
    // The raw dynamic export contains thousands of small shape/control nodes.
    // ORT Web's full optimizer can spend minutes rewriting it at startup.
    graphOptimizationLevel: "disabled",
  });
}

function validateBatch(batch: TabularBatch): ValidatedBatch {
  if (batch.x.length < 2) throw new Error("x must contain at least two rows");
  const features = batch.x[0]!.length;
  if (features < 1) throw new Error("x must contain at least one feature");
  if (batch.yTrain.length < 2 || batch.yTrain.length >= batch.x.length) {
    throw new Error("yTrain must contain at least two labels and fewer rows than x");
  }
  const flat = new Float32Array(batch.x.length * features);
  for (const [rowIndex, row] of batch.x.entries()) {
    if (row.length !== features) throw new Error(`x row ${rowIndex} has ${row.length} features; expected ${features}`);
    for (const [columnIndex, value] of row.entries()) {
      if (typeof value !== "number") throw new Error(`x[${rowIndex}][${columnIndex}] is not numeric`);
      flat[rowIndex * features + columnIndex] = value;
    }
  }
  const labels = new Set<number>();
  const y = new Float32Array(batch.yTrain.length);
  for (const [index, label] of batch.yTrain.entries()) {
    if (!Number.isInteger(label) || label < 0) throw new Error(`yTrain[${index}] must be a non-negative integer`);
    labels.add(label);
    y[index] = label;
  }
  const classes = Math.max(...labels) + 1;
  for (let label = 0; label < classes; label += 1) {
    if (!labels.has(label)) throw new Error(`training labels must be dense 0..C-1; missing ${label}`);
  }
  return { x: flat, y, totalRows: batch.x.length, trainRows: y.length, features, classes };
}

export function softmax(values: readonly number[]): number[] {
  if (values.length === 0) throw new Error("softmax requires at least one value");
  const maximum = Math.max(...values);
  const exponentials = values.map((value) => Math.exp(value - maximum));
  const denominator = exponentials.reduce((sum, value) => sum + value, 0);
  if (!Number.isFinite(denominator) || denominator <= 0) throw new Error("softmax denominator is invalid");
  return exponentials.map((value) => value / denominator);
}

function argmax(values: readonly number[]): number {
  let bestIndex = 0;
  for (let index = 1; index < values.length; index += 1) {
    if (values[index]! > values[bestIndex]!) bestIndex = index;
  }
  return bestIndex;
}
