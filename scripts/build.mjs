import { copyFile, mkdir, readFile, readdir, stat } from "node:fs/promises";
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
const modelSource = await readFile(resolve(output, "models.ts"), "utf8");
const declarations = [...modelSource.matchAll(/model\("(classification|regression)", "(fp32|int4|int8)", "([0-9a-f]{8})", ([0-9_]+)\)/gu)];
if (declarations.length !== 6) throw new Error(`Expected six model declarations, found ${declarations.length}`);
for (const [task, precisions] of [["classification", ["fp32", "int4", "int8"]], ["regression", ["fp32", "int4", "int8"]]]) {
  for (const precision of precisions) {
    const matches = modelEntries.filter((name) => new RegExp(`^tabpfn-v2-${task}-${precision}-[0-9a-f]{8}\\.onnx$`, "u").test(name));
    if (matches.length !== 1) throw new Error(`Expected one ${task}/${precision} model, found: ${matches.join(", ")}`);
    const declaration = declarations.find((match) => match[1] === task && match[2] === precision);
    if (declaration === undefined) throw new Error(`Missing ${task}/${precision} model declaration`);
    const expectedName = `tabpfn-v2-${task}-${precision}-${declaration[3]}.onnx`;
    if (matches[0] !== expectedName) throw new Error(`Stale ${task}/${precision} filename: ${expectedName}`);
    const expectedBytes = Number.parseInt(declaration[4].replaceAll("_", ""), 10);
    const actualBytes = (await stat(resolve(modelDirectory, expectedName))).size;
    if (actualBytes !== expectedBytes) throw new Error(`Stale ${task}/${precision} size: ${expectedBytes} != ${actualBytes}`);
  }
}
const library = await readFile(resolve(output, "webtabpfn.js"), "utf8");
for (const forbidden of [/\bsrc\/[^"']+\.ts\b/u, /file:\/\//u]) {
  if (forbidden.test(library)) throw new Error(`Forbidden browser reference: ${forbidden}`);
}
const packageFiles = [
  resolve("LICENSE"),
  resolve("README.md"),
  resolve("package.json"),
  ...required.map((name) => resolve(output, name)),
  ...modelEntries.filter((name) => /-(int4|int8)-/u.test(name)).map((name) => resolve(modelDirectory, name)),
];
const packageBytes = (await Promise.all(packageFiles.map(async (path) => (await stat(path)).size))).reduce((sum, size) => sum + size, 0);
if (packageBytes >= 100_000_000) throw new Error(`npm package content exceeds 100 MB: ${packageBytes} bytes`);
console.log(`Built WebTabPFN; verified models and ${packageBytes} bytes of npm package content.`);
