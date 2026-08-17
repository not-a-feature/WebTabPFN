from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge native and browser WebTabPFN benchmark JSON files")
    parser.add_argument("--task", choices=("classification", "regression"), required=True)
    parser.add_argument("--base", type=Path, help="Preserve runs from an existing consolidated result.")
    parser.add_argument("--native", action="append", default=[], metavar="LABEL=PATH")
    parser.add_argument("--browser", action="append", default=[], metavar="LABEL=PATH")
    parser.add_argument("--output", type=Path, default=Path("benchmark/classification/results.json"))
    return parser.parse_args()


def labeled_files(values: list[str]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for value in values:
        label, separator, raw_path = value.partition("=")
        assert separator == "=" and label and raw_path, value
        path = Path(raw_path).resolve(strict=True)
        run = json.loads(path.read_text(encoding="utf-8"))
        assert run["schemaVersion"] == 2
        output[label] = run
    return output


def main() -> None:
    args = parse_args()
    native: dict[str, Any] = {}
    browser: dict[str, Any] = {}
    if args.base is not None:
        base = json.loads(args.base.resolve(strict=True).read_text(encoding="utf-8"))
        assert base["schemaVersion"] == 2
        assert base["task"] == args.task
        native.update(base["native"])
        browser.update(base["browser"])
    native.update(labeled_files(args.native))
    browser.update(labeled_files(args.browser))
    assert native or browser
    for run in (*native.values(), *browser.values()):
        assert run["task"] == args.task
    payload = {
        "schemaVersion": 2,
        "createdAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "task": args.task,
        "native": native,
        "browser": browser,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
