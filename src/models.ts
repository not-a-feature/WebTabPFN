export type Task = "classification" | "regression";
export type Backend = "webgpu" | "wasm";
export type Precision = "int4" | "int8";
export type RepositoryPrecision = "fp32" | Precision;

export interface ModelSpec {
  readonly id: string;
  readonly task: Task;
  readonly file: string;
  readonly precision: RepositoryPrecision;
  readonly bytes: number;
}

const catalog: Readonly<Record<Task, Readonly<Record<RepositoryPrecision, ModelSpec>>>> = Object.freeze({
  classification: Object.freeze({
    fp32: model("classification", "fp32", "9557c06d", 29_454_872),
    int4: model("classification", "int4", "1b9a8582", 5_126_894),
    int8: model("classification", "int8", "8c5a7fbf", 8_046_373),
  }),
  regression: Object.freeze({
    fp32: model("regression", "fp32", "26f7f49c", 44_825_959),
    int4: model("regression", "int4", "56d5d4f3", 7_561_856),
    int8: model("regression", "int8", "05b4ebe5", 11_947_285),
  }),
});

export const models = Object.freeze({
  classification: Object.freeze({ int4: catalog.classification.int4, int8: catalog.classification.int8 }),
  regression: Object.freeze({ int4: catalog.regression.int4, int8: catalog.regression.int8 }),
});

export function hasWebGpu(): boolean {
  return typeof navigator !== "undefined" && "gpu" in navigator;
}

export function selectModel(task: Task, precision: RepositoryPrecision): ModelSpec {
  return catalog[task][precision];
}

function model(task: Task, precision: RepositoryPrecision, digest: string, bytes: number): ModelSpec {
  return Object.freeze({
    id: `tabpfn-v2-${task}-${precision}`,
    task,
    file: `models/tabpfn-v2-${task}-${precision}-${digest}.onnx`,
    precision,
    bytes,
  });
}
