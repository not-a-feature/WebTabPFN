from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
import tempfile
import time
from importlib.metadata import distribution
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
from export_tabpfn import configs, export
from export_tabpfn.tabpfn_patched import load_real_model
from onnx import helper, numpy_helper
from onnxruntime.quantization import QuantType, quant_utils, quantize_dynamic
from onnxruntime.quantization.matmul_nbits_quantizer import (
    DefaultWeightOnlyQuantConfig,
    MatMulNBitsQuantizer,
)
from tabpfn.constants import ModelVersion
from tabpfn.model_loading import download_model

LICENSE_URL = "https://github.com/PriorLabs/TabPFN/blob/main/LICENSE"
ATTRIBUTION = "Built with PriorLabs-TabPFN"

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export and quantize the TabPFN v2 classifier")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".build/quantization"),
    )
    parser.add_argument(
        "--publish-dir",
        type=Path,
        default=Path("src/models"),
        help="Copy ready-to-serve optimized model artifacts here.",
    )
    parser.add_argument("--no-publish", action="store_true")
    parser.add_argument("--skip-int8", action="store_true")
    parser.add_argument("--skip-int4", action="store_true")
    parser.add_argument(
        "--int4-block-size",
        type=int,
        choices=(16, 32, 64, 128, 256),
        default=32,
    )
    parser.add_argument("--skip-export-parity", action="store_true")
    parser.add_argument(
        "--reuse-fp32",
        action="store_true",
        help="Reuse an existing FP32 artifact after validating it.",
    )
    return parser.parse_args()


def resolve_checkpoint(checkpoint: Path | None, output: Path) -> Path:
    if checkpoint is not None:
        resolved = checkpoint.resolve(strict=True)
        assert resolved.is_file()
        return resolved
    target = output / "source" / "tabpfn-v2-classifier.ckpt"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        download_model(
            to=target,
            version=ModelVersion.V2,
            which="classifier",
            model_name=target.name,
        )
    resolved = target.resolve(strict=True)
    assert resolved.stat().st_size > 20_000_000
    return resolved


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def probabilities_from_logits(logits: np.ndarray, classes: int) -> np.ndarray:
    assert logits.ndim == 2
    assert 2 <= classes <= logits.shape[1]
    selected = logits[:, :classes].astype(np.float64)
    selected -= selected.max(axis=1, keepdims=True)
    exponentials = np.exp(selected)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def parity_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    assert reference.shape == candidate.shape
    assert reference.ndim == 2
    difference = np.abs(reference - candidate)
    return {
        "prediction_agreement": float(
            np.mean(np.argmax(reference, axis=1) == np.argmax(candidate, axis=1))
        ),
        "probability_mae": float(np.mean(difference)),
        "probability_max_abs": float(np.max(difference)),
    }


def export_fp32(checkpoint: Path, output: Path, check_parity: bool) -> tuple[Path, dict[str, Any]]:
    cfg = configs.real()
    model = load_real_model("classification", str(checkpoint), arch="v2")
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="webtabpfn-export-") as temporary:
        raw = Path(temporary) / "model.onnx"
        export.export_graph(
            model,
            raw,
            example=cfg.example,
            max_classes=cfg.max_classes,
            task="classification",
        )
        proto = onnx.load(str(raw), load_external_data=True)
        fp32_path = output / "tabpfn-v2-classifier-fp32.onnx"
        onnx.save_model(proto, str(fp32_path), save_as_external_data=False)
    onnx.checker.check_model(str(fp32_path), full_check=True)
    report: dict[str, Any] = {
        "export_seconds": time.perf_counter() - started,
        "bytes": fp32_path.stat().st_size,
    }
    if check_parity:
        report["export_parity"] = export.check_parity(
            fp32_path,
            model,
            cfg.parity_shapes,
            max_classes=cfg.max_classes,
            task="classification",
        )
        assert report["export_parity"]["ok"], report["export_parity"]
    return fp32_path, report


def _clear_intermediate_value_info(graph: onnx.GraphProto) -> None:
    graph.ClearField("value_info")
    for node in graph.node:
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                _clear_intermediate_value_info(attribute.g)
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                for child_graph in attribute.graphs:
                    _clear_intermediate_value_info(child_graph)


def quantize_int8(fp32_path: Path, output: Path) -> Path:
    target = output / "tabpfn-v2-classifier-int8.onnx"
    with tempfile.TemporaryDirectory(prefix="webtabpfn-int8-") as temporary:
        clean_input = _shape_inference_ready_copy(fp32_path, Path(temporary))
        quantize_dynamic(
            model_input=clean_input,
            model_output=target,
            per_channel=True,
            reduce_range=False,
            weight_type=QuantType.QInt8,
        )
    onnx.checker.check_model(str(target), full_check=True)
    return target


def quantize_int4(fp32_path: Path, output: Path, block_size: int) -> Path:
    target = output / "tabpfn-v2-classifier-int4.onnx"
    algorithm = DefaultWeightOnlyQuantConfig(
        block_size=block_size,
        is_symmetric=True,
        accuracy_level=4,
        quant_format=quant_utils.QuantFormat.QOperator,
        op_types_to_quantize=("MatMul", "Gather"),
        quant_axes=(("MatMul", 0), ("Gather", 1)),
    )
    with tempfile.TemporaryDirectory(prefix="webtabpfn-int4-") as temporary:
        clean_input = _shape_inference_ready_copy(fp32_path, Path(temporary))
        model = quant_utils.load_model_with_shape_infer(clean_input)
        quantizer = MatMulNBitsQuantizer(
            model,
            nodes_to_exclude=None,
            nodes_to_include=None,
            algo_config=algorithm,
        )
        quantizer.process()
        quantizer.model.save_model_to_file(str(target), False)
    onnx.checker.check_model(str(target), full_check=True)
    return target


def _shape_inference_ready_copy(model_path: Path, directory: Path) -> Path:
    """Remove stale exporter value_info before ORT quantizer shape inference."""
    model = onnx.load(str(model_path))
    _canonicalize_constant_weight_linears(model.graph)
    _clear_intermediate_value_info(model.graph)
    clean_path = directory / model_path.name
    onnx.save_model(model, str(clean_path), save_as_external_data=False)
    return clean_path


def _canonicalize_constant_weight_linears(graph: onnx.GraphProto) -> None:
    """Expose exported linear weights as constant MatMul B inputs.

    torch.export emits most linear weights behind a Transpose and the MLP
    layers as Gemm. ORT's N-bit quantizer only packs constant MatMul/Gather
    inputs, so both forms must be canonicalized before quantization.
    """
    initializers = {initializer.name: initializer for initializer in graph.initializer}
    folded_nodes: list[onnx.NodeProto] = []
    new_initializers: list[onnx.TensorProto] = []

    for node in graph.node:
        if node.op_type == "Transpose" and node.input[0] in initializers:
            attributes = {
                attribute.name: helper.get_attribute_value(attribute)
                for attribute in node.attribute
            }
            array = numpy_helper.to_array(initializers[node.input[0]])
            permutation = attributes.get("perm")
            transposed = np.transpose(array, axes=permutation)
            initializer = numpy_helper.from_array(
                np.ascontiguousarray(transposed),
                name=node.output[0],
            )
            new_initializers.append(initializer)
            initializers[initializer.name] = initializer
        else:
            folded_nodes.append(node)

    rewritten_nodes: list[onnx.NodeProto] = []
    for node in folded_nodes:
        if node.op_type != "Gemm" or node.input[1] not in initializers:
            rewritten_nodes.append(node)
            continue
        attributes = {
            attribute.name: helper.get_attribute_value(attribute)
            for attribute in node.attribute
        }
        if int(attributes.get("transA", 0)) != 0:
            rewritten_nodes.append(node)
            continue

        weight = numpy_helper.to_array(initializers[node.input[1]])
        if int(attributes.get("transB", 0)):
            weight = weight.T
        weight = weight * float(attributes.get("alpha", 1.0))
        weight_name = f"{node.name}_weight_for_matmul"
        weight_initializer = numpy_helper.from_array(
            np.ascontiguousarray(weight),
            name=weight_name,
        )
        new_initializers.append(weight_initializer)
        initializers[weight_name] = weight_initializer

        has_bias = len(node.input) > 2 and bool(node.input[2])
        matmul_output = f"{node.output[0]}_before_bias" if has_bias else node.output[0]
        rewritten_nodes.append(
            helper.make_node(
                "MatMul",
                [node.input[0], weight_name],
                [matmul_output],
                name=f"{node.name}_MatMul",
            )
        )
        if has_bias:
            bias_name = node.input[2]
            beta = float(attributes.get("beta", 1.0))
            if beta != 1.0 and bias_name in initializers:
                bias = numpy_helper.to_array(initializers[bias_name]) * beta
                bias_name = f"{node.name}_scaled_bias"
                bias_initializer = numpy_helper.from_array(
                    np.ascontiguousarray(bias),
                    name=bias_name,
                )
                new_initializers.append(bias_initializer)
                initializers[bias_name] = bias_initializer
            rewritten_nodes.append(
                helper.make_node(
                    "Add",
                    [matmul_output, bias_name],
                    list(node.output),
                    name=node.name,
                )
            )

    graph.ClearField("node")
    graph.node.extend(rewritten_nodes)
    graph.initializer.extend(new_initializers)

    used_names = {name for node in graph.node for name in node.input if name}
    used_names.update(value.name for value in graph.input)
    retained = [
        initializer
        for initializer in graph.initializer
        if initializer.name in used_names
    ]
    graph.ClearField("initializer")
    graph.initializer.extend(retained)


def compare_variants(fp32_path: Path, candidates: dict[str, Path]) -> dict[str, Any]:
    rng = np.random.default_rng(20260817)
    shapes = ((32, 12, 20), (64, 20, 40))
    cases: list[tuple[tuple[int, int, int], dict[str, np.ndarray]]] = []
    for total_rows, features, train_rows in shapes:
        x = rng.normal(size=(1, total_rows, features)).astype(np.float32)
        y = (np.arange(train_rows) % 4).astype(np.float32)[None, :]
        cases.append(((total_rows, features, train_rows), {"x": x, "y": y}))
    reference_session = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    reports: dict[str, Any] = {}
    for name, path in candidates.items():
        candidate_session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        per_shape: list[dict[str, Any]] = []
        for (total_rows, features, train_rows), feeds in cases:
            reference_logits = reference_session.run(["logits"], feeds)[0][0, train_rows:, :]
            candidate_logits = candidate_session.run(["logits"], feeds)[0][0, train_rows:, :]
            reference_probabilities = probabilities_from_logits(reference_logits, classes=4)
            candidate_probabilities = probabilities_from_logits(candidate_logits, classes=4)
            per_shape.append(
                {
                    "shape": [total_rows, features, train_rows],
                    **parity_metrics(reference_probabilities, candidate_probabilities),
                }
            )
        reports[name] = per_shape
    return reports


def optimize_for_web(source: Path, output: Path) -> Path:
    """Apply portable graph optimization while retaining browser-compatible ONNX."""
    target = output / f"{source.stem}-basic.onnx"
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    options.optimized_model_filepath = str(target)
    ort.InferenceSession(str(source), options, providers=["CPUExecutionProvider"])
    onnx.checker.check_model(str(target), full_check=True)
    assert target.stat().st_size > 1_000_000
    return target


def make_manifest(checkpoint: Path, variants: dict[str, tuple[Path, list[str]]]) -> dict[str, Any]:
    payload: list[dict[str, Any]] = []
    for precision, (path, providers) in variants.items():
        payload.append(
            {
                "id": f"v2-classifier-{precision}-{path.suffix[1:]}",
                "url": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "precision": precision,
                "providers": providers,
            }
        )
    return {
        "schemaVersion": 1,
        "modelId": "tabpfn-v2-classifier",
        "checkpointSha256": sha256_file(checkpoint),
        "maxClasses": 10,
        "license": {
            "name": "Prior Labs License (Apache 2.0 with additional provision)",
            "url": LICENSE_URL,
            "attribution": ATTRIBUTION,
        },
        "variants": payload,
    }


def publish_release(
    publish_dir: Path,
    variants: dict[str, tuple[Path, list[str]]],
) -> None:
    publish_dir.mkdir(parents=True, exist_ok=True)
    for precision, (path, _providers) in variants.items():
        for old_model in publish_dir.glob(f"tabpfn-v2-classifier-{precision}-*.onnx"):
            old_model.unlink()
        digest = sha256_file(path)[:8]
        shutil.copy2(path, publish_dir / f"tabpfn-v2-classifier-{precision}-{digest}.onnx")
    tabpfn_distribution = distribution("tabpfn")
    license_file = next(
        (
            file
            for file in (tabpfn_distribution.files or [])
            if str(file).replace("\\", "/").endswith("licenses/LICENSE")
        ),
        None,
    )
    if license_file is None:
        raise FileNotFoundError("Installed tabpfn distribution does not contain its LICENSE")
    shutil.copy2(
        tabpfn_distribution.locate_file(license_file),
        publish_dir.parent / "TABPFN_MODEL_LICENSE.txt",
    )


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = resolve_checkpoint(args.checkpoint, output)
    fp32_path = output / "tabpfn-v2-classifier-fp32.onnx"
    if args.reuse_fp32:
        if not fp32_path.exists():
            raise FileNotFoundError(f"Cannot reuse missing artifact: {fp32_path}")
        onnx.checker.check_model(str(fp32_path), full_check=True)
        export_report = {
            "reused": True,
            "bytes": fp32_path.stat().st_size,
        }
    else:
        fp32_path, export_report = export_fp32(
            checkpoint,
            output,
            not args.skip_export_parity,
        )
    logging.getLogger().setLevel(logging.WARNING)
    logging.getLogger("onnxruntime.quantization.matmul_nbits_quantizer").setLevel(logging.WARNING)
    fp32_web_path = optimize_for_web(fp32_path, output)
    variants: dict[str, tuple[Path, list[str]]] = {
        "fp32": (fp32_web_path, ["webgpu", "wasm"]),
    }
    compared: dict[str, Path] = {"fp32": fp32_web_path}
    development_onnx: dict[str, Path] = {"fp32": fp32_path}
    if not args.skip_int8:
        int8_path = quantize_int8(fp32_path, output)
        int8_web_path = optimize_for_web(int8_path, output)
        variants["int8"] = (int8_web_path, ["webgpu", "wasm"])
        compared["int8"] = int8_web_path
        development_onnx["int8"] = int8_path
    if not args.skip_int4:
        int4_path = quantize_int4(fp32_path, output, args.int4_block_size)
        int4_web_path = optimize_for_web(int4_path, output)
        variants["int4"] = (int4_web_path, ["webgpu", "wasm"])
        compared["int4"] = int4_web_path
        development_onnx["int4"] = int4_path
    manifest = make_manifest(checkpoint, variants)
    write_json(output / "manifest.json", manifest)
    if not args.no_publish:
        publish_release(args.publish_dir.resolve(), variants)
    report = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": manifest["checkpointSha256"],
        "export": export_report,
        "sizes": {
            precision: {
                "bytes": path.stat().st_size,
                "ratio_to_fp32": path.stat().st_size / fp32_path.stat().st_size,
            }
            for precision, (path, _providers) in variants.items()
        },
        "development_onnx_sizes": {
            precision: path.stat().st_size
            for precision, path in development_onnx.items()
        },
        "int4": {
            "bits": 4,
            "block_size": args.int4_block_size,
            "symmetric": True,
            "format": "MatMulNBits",
        },
        "native_web_parity": compare_variants(fp32_path, compared),
    }
    write_json(output / "quantization-report.json", report)


if __name__ == "__main__":
    main()
