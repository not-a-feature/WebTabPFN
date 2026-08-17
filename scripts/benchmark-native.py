from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import sklearn
import tabpfn
import torch
from sklearn.datasets import (
    load_breast_cancer,
    load_diabetes,
    load_digits,
    load_iris,
    load_wine,
    make_circles,
    make_classification,
    make_moons,
    make_regression,
)
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    log_loss,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn.constants import ModelVersion

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate regular-Python TabPFN v2 benchmark references")
    parser.add_argument("--task", choices=("classification", "regression"), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-estimators", type=int, default=1)
    parser.add_argument("--train-rows", type=int, default=128)
    parser.add_argument("--test-rows", type=int, default=48)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    return parser.parse_args()


def datasets(seed: int, task: str) -> list[tuple[str, np.ndarray, np.ndarray]]:
    if task == "regression":
        diabetes = load_diabetes()
        synthetic_x, synthetic_y = make_regression(
            n_samples=640,
            n_features=20,
            n_informative=12,
            noise=8.0,
            random_state=seed,
        )
        offset_x, offset_y = make_regression(
            n_samples=640,
            n_features=12,
            n_informative=8,
            noise=20.0,
            bias=10_000.0,
            random_state=seed + 1,
        )
        return [
            ("diabetes", diabetes.data.astype(np.float32), diabetes.target.astype(np.float32)),
            ("synthetic", synthetic_x.astype(np.float32), synthetic_y.astype(np.float32)),
            ("synthetic_offset", offset_x.astype(np.float32), offset_y.astype(np.float32)),
        ]
    breast = load_breast_cancer()
    digits = load_digits()
    iris = load_iris()
    wine = load_wine()
    synthetic_x, synthetic_y = make_classification(
        n_samples=640,
        n_features=24,
        n_informative=12,
        n_redundant=4,
        n_classes=4,
        n_clusters_per_class=1,
        class_sep=1.2,
        random_state=seed,
    )
    imbalanced_x, imbalanced_y = make_classification(
        n_samples=800,
        n_features=32,
        n_informative=18,
        n_redundant=6,
        n_classes=3,
        n_clusters_per_class=1,
        weights=[0.7, 0.2, 0.1],
        class_sep=1.0,
        random_state=seed + 1,
    )
    moons_x, moons_y = make_moons(n_samples=640, noise=0.25, random_state=seed + 2)
    circles_x, circles_y = make_circles(
        n_samples=640,
        factor=0.45,
        noise=0.12,
        random_state=seed + 3,
    )
    return [
        ("breast_cancer", breast.data.astype(np.float32), breast.target.astype(np.int64)),
        ("wine", wine.data.astype(np.float32), wine.target.astype(np.int64)),
        ("synthetic_4class", synthetic_x.astype(np.float32), synthetic_y.astype(np.int64)),
        ("iris", iris.data.astype(np.float32), iris.target.astype(np.int64)),
        ("digits", digits.data.astype(np.float32), digits.target.astype(np.int64)),
        ("moons", moons_x.astype(np.float32), moons_y.astype(np.int64)),
        ("circles", circles_x.astype(np.float32), circles_y.astype(np.int64)),
        (
            "synthetic_imbalanced_3class",
            imbalanced_x.astype(np.float32),
            imbalanced_y.astype(np.int64),
        ),
    ]


def split_dataset(
    x: np.ndarray,
    y: np.ndarray,
    train_rows: int,
    test_rows: int,
    seed: int,
    task: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    assert x.ndim == 2
    assert y.ndim == 1
    assert x.shape[0] == y.shape[0]
    assert train_rows + test_rows <= x.shape[0]
    splitter = (StratifiedShuffleSplit if task == "classification" else ShuffleSplit)(
        n_splits=1,
        train_size=train_rows,
        test_size=test_rows,
        random_state=seed,
    )
    train_indices, test_indices = next(splitter.split(x, y if task == "classification" else None))
    return x[train_indices], x[test_indices], y[train_indices], y[test_indices]


def make_estimator(args: argparse.Namespace) -> TabPFNClassifier | TabPFNRegressor:
    assert args.n_estimators >= 1
    common = {
        "device": args.device,
        "n_estimators": args.n_estimators,
        "random_state": args.seed,
        "auto_scale_n_estimators": False,
    }
    if args.checkpoint is not None:
        checkpoint = args.checkpoint.resolve(strict=True)
        estimator = TabPFNClassifier if args.task == "classification" else TabPFNRegressor
        return estimator(model_path=checkpoint, **common)
    estimator = TabPFNClassifier if args.task == "classification" else TabPFNRegressor
    return estimator.create_default_for_version(version=ModelVersion.V2, **common)


def evaluate_case(
    estimator: TabPFNClassifier | TabPFNRegressor,
    task: str,
    name: str,
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    runs: int,
    warmups: int,
    device: str,
) -> dict[str, Any]:
    assert runs >= 1
    assert warmups >= 0
    rss_samples = [rss_bytes()]
    for _ in range(warmups):
        estimator.fit(x_train, y_train)
        if task == "classification":
            assert isinstance(estimator, TabPFNClassifier)
            estimator.predict_proba(x_test)
        else:
            assert isinstance(estimator, TabPFNRegressor)
            estimator.predict(x_test)
        synchronize(device)

    fit_timings: list[float] = []
    predict_timings: list[float] = []
    output: np.ndarray | None = None
    for _ in range(runs):
        synchronize(device)
        fit_started = time.perf_counter()
        estimator.fit(x_train, y_train)
        synchronize(device)
        fit_timings.append((time.perf_counter() - fit_started) * 1000)
        rss_samples.append(rss_bytes())

        predict_started = time.perf_counter()
        if task == "classification":
            assert isinstance(estimator, TabPFNClassifier)
            output = estimator.predict_proba(x_test)
        else:
            assert isinstance(estimator, TabPFNRegressor)
            output = estimator.predict(x_test)
        synchronize(device)
        predict_timings.append((time.perf_counter() - predict_started) * 1000)
        rss_samples.append(rss_bytes())

    assert output is not None
    result: dict[str, Any] = {
        "name": name,
        "x": np.concatenate([x_train, x_test], axis=0).tolist(),
        "yTrain": y_train.tolist(),
        "yTest": y_test.tolist(),
    }
    if task == "classification":
        assert output.shape == (x_test.shape[0], np.unique(y_train).size)
        predictions = np.argmax(output, axis=1)
        result["python"] = {
            "probabilities": output.tolist(),
            "predictions": predictions.tolist(),
            "fitMs": float(np.median(fit_timings)),
            "predictMs": float(np.median(predict_timings)),
            "timingMs": {
                "fit": timing_summary(fit_timings),
                "predict": timing_summary(predict_timings),
            },
            "sampledPeakRssMb": max(rss_samples) / (1024 * 1024),
            "metrics": {
                "accuracy": float(accuracy_score(y_test, predictions)),
                "balancedAccuracy": float(balanced_accuracy_score(y_test, predictions)),
                "logLoss": float(log_loss(y_test, output, labels=np.arange(output.shape[1]))),
                "mcc": float(matthews_corrcoef(y_test, predictions)),
            },
        }
    else:
        assert output.shape == (x_test.shape[0],)
        result["python"] = {
            "predictions": output.tolist(),
            "fitMs": float(np.median(fit_timings)),
            "predictMs": float(np.median(predict_timings)),
            "timingMs": {
                "fit": timing_summary(fit_timings),
                "predict": timing_summary(predict_timings),
            },
            "sampledPeakRssMb": max(rss_samples) / (1024 * 1024),
            "metrics": {
                "mae": float(mean_absolute_error(y_test, output)),
                "rmse": float(np.sqrt(mean_squared_error(y_test, output))),
                "r2": float(r2_score(y_test, output)),
            },
        }
    return result


def synchronize(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize(torch.device(device))


def timing_summary(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "runs": values,
        "count": len(values),
        "min": float(np.min(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "p50": float(np.percentile(array, 50)),
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(np.max(array)),
    }


def rss_bytes() -> int:
    return int(psutil.Process().memory_info().rss)


def total_memory_bytes() -> int:
    return int(psutil.virtual_memory().total)


def environment(device: str, n_estimators: int) -> dict[str, Any]:
    value: dict[str, Any] = {
        "hostname": "redacted",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpuLogical": os.cpu_count(),
        "cpuPhysical": psutil.cpu_count(logical=False),
        "memoryGb": total_memory_bytes() / (1024**3),
        "tabpfn": tabpfn.__version__,
        "sklearn": sklearn.__version__,
        "torch": torch.__version__,
        "device": device,
        "nEstimators": n_estimators,
    }
    if device.startswith("cuda"):
        index = torch.device(device).index
        assert index is not None
        properties = torch.cuda.get_device_properties(index)
        value["gpu"] = {
            "index": index,
            "name": properties.name,
            "totalMemoryGb": properties.total_memory / (1024**3),
            "computeCapability": f"{properties.major}.{properties.minor}",
        }
    return value


def main() -> None:
    args = parse_args()
    assert args.train_rows >= 16
    assert args.test_rows >= 8
    assert args.runs >= 1
    assert args.warmups >= 0
    create_started = time.perf_counter()
    estimator = make_estimator(args)
    estimator_create_ms = (time.perf_counter() - create_started) * 1000
    cases: list[dict[str, Any]] = []
    for index, (name, x, y) in enumerate(datasets(args.seed, args.task)):
        case_test_rows = min(args.test_rows, x.shape[0] - 16)
        case_train_rows = min(args.train_rows, x.shape[0] - case_test_rows)
        x_train, x_test, y_train, y_test = split_dataset(
            x,
            y,
            case_train_rows,
            case_test_rows,
            args.seed + index,
            args.task,
        )
        cases.append(
            evaluate_case(
                estimator,
                args.task,
                name,
                x_train,
                x_test,
                y_train,
                y_test,
                args.runs,
                args.warmups,
                args.device,
            )
        )
    payload = {
        "schemaVersion": 2,
        "createdAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "task": args.task,
        "modelId": f"tabpfn-v2-{args.task}",
        "checkpointSha256": (
            sha256_file(args.checkpoint.resolve(strict=True)) if args.checkpoint is not None else None
        ),
        "environment": environment(args.device, args.n_estimators),
        "benchmark": {"runs": args.runs, "warmups": args.warmups},
        "estimatorCreateMs": estimator_create_ms,
        "cases": cases,
    }
    output = args.output or Path(f"benchmark/{args.task}/reference.json")
    write_json(output.resolve(), payload)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
