import * as ort from "onnxruntime-web/webgpu";

import type { Backend, Task } from "./models";

export interface TabularBatch {
  readonly x: readonly (readonly number[])[];
  readonly yTrain: readonly number[];
}

export interface InferenceInfo {
  readonly backend: Backend;
  readonly inferenceMs: number;
}

export interface EncodedClassificationResult extends InferenceInfo {
  readonly probabilities: readonly (readonly number[])[];
  readonly predictions: readonly number[];
}

export interface RegressionResult extends InferenceInfo {
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

interface RawOutput {
  readonly input: ValidatedBatch;
  readonly logits: Float32Array;
  readonly columns: number;
  readonly inferenceMs: number;
}

export class TabPFNSession {
  readonly backend: Backend;
  readonly task: Task;
  readonly #session: ort.InferenceSession;

  private constructor(
    session: ort.InferenceSession,
    backend: Backend,
    task: Task,
  ) {
    this.#session = session;
    this.backend = backend;
    this.task = task;
  }

  static async create(
    model: Uint8Array,
    backend: Backend,
    task: Task,
    runtimeBaseUrl: string,
  ): Promise<TabPFNSession> {
    ort.env.wasm.numThreads = 0;
    ort.env.wasm.wasmPaths = {
      mjs: new URL("ort-wasm-simd-threaded.asyncify.mjs", runtimeBaseUrl).href,
      wasm: new URL("ort-wasm-simd-threaded.asyncify.wasm", runtimeBaseUrl).href,
    };
    if (backend === "wasm") {
      return new TabPFNSession(await createSession(model, ["wasm"]), "wasm", task);
    }
    if (!("gpu" in navigator)) throw new Error("WebGPU was requested but navigator.gpu is unavailable");
    return new TabPFNSession(await createSession(model, ["webgpu"]), "webgpu", task);
  }

  async dispose(): Promise<void> {
    await this.#session.release();
  }

  async predictClassification(batch: TabularBatch): Promise<EncodedClassificationResult> {
    if (this.task !== "classification") throw new Error("classification called with a regression model");
    const output = await this.#run(batch);
    if (output.columns < output.input.classes) throw new Error("model output has fewer classes than the training labels");

    const probabilities: number[][] = [];
    const predictions: number[] = [];
    for (let row = output.input.trainRows; row < output.input.totalRows; row += 1) {
      const offset = row * output.columns;
      const probability = softmax(Array.from(output.logits.slice(offset, offset + output.input.classes)));
      probabilities.push(probability);
      predictions.push(argmax(probability));
    }
    return { ...this.#info(output.inferenceMs), probabilities, predictions };
  }

  async predictRegression(batch: TabularBatch): Promise<RegressionResult> {
    if (this.task !== "regression") throw new Error("regression called with a classification model");
    const output = await this.#run(batch);
    if (output.columns !== 1) throw new Error(`regression model returned ${output.columns} columns; expected 1`);
    const predictions: number[] = [];
    for (let row = output.input.trainRows; row < output.input.totalRows; row += 1) {
      const value = output.logits[row]!;
      if (!Number.isFinite(value)) throw new Error(`regression prediction ${row - output.input.trainRows} is not finite`);
      predictions.push(value);
    }
    return { ...this.#info(output.inferenceMs), predictions };
  }

  async #run(batch: TabularBatch): Promise<RawOutput> {
    const input = validateBatch(batch, this.task);
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
    return { input, logits: logits.data, columns: logits.dims[2]!, inferenceMs };
  }

  #info(inferenceMs: number): InferenceInfo {
    return { backend: this.backend, inferenceMs };
  }
}

async function createSession(model: Uint8Array, executionProviders: readonly Backend[]): Promise<ort.InferenceSession> {
  return ort.InferenceSession.create(model, {
    executionProviders: [...executionProviders],
    graphOptimizationLevel: "disabled",
  });
}

function validateBatch(batch: TabularBatch, task: Task): ValidatedBatch {
  if (batch.x.length < 2) throw new Error("x must contain at least two rows");
  const features = batch.x[0]!.length;
  if (features < 1) throw new Error("x must contain at least one feature");
  if (batch.yTrain.length < 2 || batch.yTrain.length >= batch.x.length) {
    throw new Error("yTrain must contain at least two targets and fewer rows than x");
  }
  const flat = new Float32Array(batch.x.length * features);
  for (const [rowIndex, row] of batch.x.entries()) {
    if (row.length !== features) throw new Error(`x row ${rowIndex} has ${row.length} features; expected ${features}`);
    for (const [columnIndex, value] of row.entries()) {
      if (typeof value !== "number") throw new Error(`x[${rowIndex}][${columnIndex}] is not numeric`);
      if (!Number.isFinite(value) && !Number.isNaN(value)) throw new Error(`x[${rowIndex}][${columnIndex}] must not be infinite`);
      flat[rowIndex * features + columnIndex] = value;
    }
  }

  const y = new Float32Array(batch.yTrain.length);
  let classes = 0;
  if (task === "classification") {
    const labels = new Set<number>();
    for (const [index, label] of batch.yTrain.entries()) {
      if (!Number.isInteger(label) || label < 0) throw new Error(`yTrain[${index}] must be a non-negative integer`);
      labels.add(label);
      y[index] = label;
    }
    classes = Math.max(...labels) + 1;
    for (let label = 0; label < classes; label += 1) {
      if (!labels.has(label)) throw new Error(`training labels must be dense 0..C-1; missing ${label}`);
    }
  } else {
    for (const [index, target] of batch.yTrain.entries()) {
      if (!Number.isFinite(target)) throw new Error(`yTrain[${index}] must be finite`);
      y[index] = target;
    }
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
