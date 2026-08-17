import { copyFile, mkdir, readFile, readdir } from "node:fs/promises";
import { resolve } from "node:path";

import { build } from "esbuild";

const output = resolve("src");
const modelDirectory = resolve(output, "models");
await mkdir(output, { recursive: true });
await mkdir(modelDirectory, { recursive: true });

await build({
  bundle: true,
  conditions: ["onnxruntime-web-use-extern-wasm"],
  entryPoints: [resolve("src", "index.ts")],
  format: "iife",
  globalName: "WebTabPFN",
  legalComments: "none",
  minify: true,
  outfile: resolve(output, "webtabpfn.js"),
  platform: "browser",
  target: "es2022",
});

const assets = [
  [resolve("node_modules", "onnxruntime-web", "dist", "ort-wasm-simd-threaded.asyncify.mjs"), "ort-wasm-simd-threaded.asyncify.mjs"],
  [resolve("node_modules", "onnxruntime-web", "dist", "ort-wasm-simd-threaded.asyncify.wasm"), "ort-wasm-simd-threaded.asyncify.wasm"],
];
await Promise.all(assets.map(([source, name]) => copyFile(source, resolve(output, name))));

const required = [
  "ONNXRUNTIME_LICENSE.txt",
  "ONNXRUNTIME_THIRD_PARTY_NOTICES.txt",
  "TABPFN_MODEL_LICENSE.txt",
  "ort-wasm-simd-threaded.asyncify.mjs",
  "ort-wasm-simd-threaded.asyncify.wasm",
  "webtabpfn.js",
];
const entries = await readdir(output);
const modelEntries = await readdir(modelDirectory);
for (const name of required) {
  if (!entries.includes(name)) throw new Error(`Missing browser asset: ${name}`);
}
for (const precision of ["fp32", "int4", "int8"]) {
  const matches = modelEntries.filter((name) => new RegExp(`^tabpfn-v2-classifier-${precision}-[0-9a-f]{8}\\.onnx$`, "u").test(name));
  if (matches.length !== 1) throw new Error(`Expected one ${precision} model, found: ${matches.join(", ")}`);
}
const library = await readFile(resolve(output, "webtabpfn.js"), "utf8");
for (const forbidden of [/\bsrc\/[^"']+\.ts\b/u, /file:\/\//u]) {
  if (forbidden.test(library)) throw new Error(`Forbidden browser reference: ${forbidden}`);
}
console.log("Built the browser library in src/ and verified all three model variants.");
