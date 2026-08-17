export type Backend = "auto" | "webgpu" | "wasm";
export type ConcreteBackend = Exclude<Backend, "auto">;
export type Precision = "fp32" | "int4" | "int8";

export interface ModelSpec {
  readonly id: string;
  readonly file: string;
  readonly precision: Precision;
  readonly bytes: number;
  readonly sha256: string;
}

export const models: Readonly<Record<Precision, ModelSpec>> = Object.freeze({
  fp32: Object.freeze({
    id: "tabpfn-v2-fp32",
    file: "models/tabpfn-v2-classifier-fp32-9557c06d.onnx",
    precision: "fp32",
    bytes: 29_454_872,
    sha256: "9557c06dbd125d75dfe7a271f701be01e05111c5425000e01dc03ef12e624a25",
  }),
  int4: Object.freeze({
    id: "tabpfn-v2-int4",
    file: "models/tabpfn-v2-classifier-int4-1b9a8582.onnx",
    precision: "int4",
    bytes: 5_126_894,
    sha256: "1b9a85824cc6caa9f1546892f37b0f47f4445ff688169bbab018912f00bbe541",
  }),
  int8: Object.freeze({
    id: "tabpfn-v2-int8",
    file: "models/tabpfn-v2-classifier-int8-8c5a7fbf.onnx",
    precision: "int8",
    bytes: 8_046_373,
    sha256: "8c5a7fbf5bb5c6e4fdbaeccbdd759cd2790f0d559c50b45a5512b95b862c6d60",
  }),
});

export function hasWebGpu(): boolean {
  return typeof navigator !== "undefined" && "gpu" in navigator;
}

export function resolveBackend(requested: Backend = "auto"): ConcreteBackend {
  return requested === "auto" ? (hasWebGpu() ? "webgpu" : "wasm") : requested;
}

export function selectModel(backend: ConcreteBackend, precision?: Precision): ModelSpec {
  return models[precision ?? (backend === "webgpu" ? "int4" : "int8")];
}
