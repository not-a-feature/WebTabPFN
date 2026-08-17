from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from xml.sax.saxutils import escape


TYPE_COLORS = {"webgpu": "#6EA6DD", "wasm": "#E9A35E", "native": "#70BD8E"}
PARETO_TYPE_COLORS = {"webgpu": "#2878B5", "wasm": "#E07A1F", "native": "#26935C"}
REFERENCE_COLOR = "#98A4B3"
FRONT_COLOR = "#D98F8F"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot WebTabPFN benchmark results as SVG")
    parser.add_argument("--input", type=Path, default=Path("benchmark/classification/results.json"))
    parser.add_argument("--output", type=Path, default=Path("benchmark/classification/plots"))
    return parser.parse_args()


def browser_rows(results: dict) -> list[dict]:
    rows = []
    task = results["task"]
    for run in results["browser"].values():
        for configuration in run["configurations"]:
            cases = configuration["cases"]
            row = {
                "label": f'browser {configuration["backend"]}/{configuration["precision"]}',
                "kind": configuration["backend"],
                "reference_label": None,
                "precision": configuration["precision"],
                "bytes": configuration["model"]["bytes"],
                "p50": mean(case["timingMs"]["p50"] for case in cases),
                "p95": mean(case["timingMs"]["p95"] for case in cases),
            }
            metrics = ("accuracy", "mcc") if task == "classification" else ("mae", "rmse", "r2")
            row.update({metric: mean(case["metrics"][metric] for case in cases) for metric in metrics})
            rows.append(row)
    return rows


def native_rows(results: dict) -> list[dict]:
    rows = []
    task = results["task"]
    for name, run in results["native"].items():
        cases = run["cases"]
        row = {
            "label": f"native {name}",
            "kind": "native",
            "reference_label": run["referenceLabel"] if "referenceLabel" in run else None,
            "p50": mean(case["python"]["timingMs"]["predict"]["p50"] for case in cases),
            "p95": mean(case["python"]["timingMs"]["predict"]["p95"] for case in cases),
        }
        metrics = ("accuracy", "mcc") if task == "classification" else ("mae", "rmse", "r2")
        row.update({metric: mean(case["python"]["metrics"][metric] for case in cases) for metric in metrics})
        rows.append(row)
    return rows


def document(title: str, width: int, height: int, body: list[str]) -> str:
    style = """
      text { font-family: ui-sans-serif, system-ui, sans-serif; fill: #172033 }
      .axis { font-size: 12px; fill: #526070 }
      .label { font-size: 13px }
      .bar-value { font-size: 11px; font-weight: 600 }
      .grid { stroke: #d9e0e8; stroke-width: 1 }
    """
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img" aria-label="{escape(title)}">\n'
        f"<style>{style}</style>\n<title>{escape(title)}</title>\n"
        + "\n".join(body)
        + "\n</svg>\n"
    )


def horizontal_plot(
    title: str,
    x_label: str,
    rows: list[dict],
    fields: tuple[tuple[str, str], ...],
    minimum: float,
    maximum: float,
    references: tuple[tuple[str, float], ...] = (),
) -> str:
    assert maximum > minimum
    for row in rows:
        for field, _label in fields:
            assert minimum <= row[field] <= maximum, (row["label"], field, row[field])
    for reference_label, reference_value in references:
        assert minimum <= reference_value <= maximum, (reference_label, reference_value)
    width, left, right, top, row_height = 960, 235, 35, 75, 46
    plot_bottom = top + len(rows) * row_height + 5
    height = plot_bottom + 58
    plot_width = width - left - right
    body: list[str] = []
    reference_overlay: list[str] = []
    for index in range(6):
        value = minimum + (maximum - minimum) * index / 5
        x = left + plot_width * index / 5
        tick = f"{value:.0f}" if maximum >= 100 else f"{value:.2f}"
        body.extend(
            (
                f'<line class="grid" x1="{x:.1f}" y1="{top - 15}" x2="{x:.1f}" y2="{plot_bottom}"/>',
                f'<text class="axis" x="{x:.1f}" y="{plot_bottom + 22}" text-anchor="middle">{tick}</text>',
            )
        )
    for reference_label, reference_value in references:
        x = left + (reference_value - minimum) / (maximum - minimum) * plot_width
        reference_overlay.append(
            f'<line x1="{x:.1f}" y1="{top - 15}" x2="{x:.1f}" y2="{plot_bottom}" stroke="{REFERENCE_COLOR}" stroke-width="3" stroke-dasharray="8 6">'
            f'<title>{escape(reference_label)}</title></line>'
        )
    bar_height = 16 if len(fields) == 2 else 24
    bar_group_height = len(fields) * bar_height + (len(fields) - 1) * 3
    for row_index, row in enumerate(rows):
        y = top + row_index * row_height
        body.append(
            f'<text class="label" x="{left - 10}" y="{y + bar_group_height / 2 + 4:.1f}" text-anchor="end">'
            f'{escape(row["label"])}</text>'
        )
        for field_index, (field, label) in enumerate(fields):
            value = row[field]
            bar_y = y + field_index * (bar_height + 3)
            bar_width = (value - minimum) / (maximum - minimum) * plot_width
            color = TYPE_COLORS[row["kind"]]
            opacity = 1.0 if field_index == 0 else 0.58
            body.append(
                f'<rect x="{left}" y="{bar_y}" width="{bar_width:.1f}" height="{bar_height}" rx="3" fill="{color}" fill-opacity="{opacity:.2f}">'
                f'<title>{escape(row["label"])} · {escape(label)}: {value:.4f}</title></rect>'
            )
            value_text = f"{value:.1f}" if maximum >= 10 else f"{value:.3f}"
            bar_end = left + bar_width
            fits_outside = bar_end + len(value_text) * 7 + 8 <= left + plot_width
            value_x = bar_end + 5 if fits_outside else bar_end - 5
            value_anchor = "start" if fits_outside else "end"
            value_style = "" if fits_outside else ' style="fill:#fff"'
            body.append(
                f'<text class="bar-value" x="{value_x:.1f}" y="{bar_y + bar_height - 4}" '
                f'text-anchor="{value_anchor}"{value_style}>{value_text}</text>'
            )
    body.extend(reference_overlay)
    legend_items = [
        (kind, label)
        for kind, label in (("webgpu", "WebGPU"), ("wasm", "WASM"), ("native", "Native"))
        if any(row["kind"] == kind for row in rows)
    ]
    legend_x = width - len(legend_items) * 100
    for index, (kind, label) in enumerate(legend_items):
        body.extend(
            (
                f'<rect x="{legend_x + index * 100}" y="24" width="12" height="12" fill="{TYPE_COLORS[kind]}"/>',
                f'<text class="axis" x="{legend_x + 17 + index * 100}" y="35">{escape(label)}</text>',
            )
        )
    for index, (reference_label, _reference_value) in enumerate(references):
        reference_x = left + index * 190
        body.extend(
            (
                f'<line x1="{reference_x}" y1="53" x2="{reference_x + 28}" y2="53" stroke="{REFERENCE_COLOR}" stroke-width="3" stroke-dasharray="8 6"/>',
                f'<text class="axis" x="{reference_x + 35}" y="57">{escape(reference_label)}</text>',
            )
        )
    if len(fields) == 2:
        body.append(
            f'<text class="axis" x="235" y="35">{escape(fields[0][1])} solid · {escape(fields[1][1])} light</text>'
        )
    body.append(
        f'<text class="axis" x="{left + plot_width / 2:.1f}" y="{height - 12}" text-anchor="middle">'
        f'{escape(x_label)}</text>'
    )
    return document(title, width, height, body)


def pareto_plot(rows: list[dict], quality: str, quality_label: str, title: str) -> str:
    ordered = sorted(rows, key=lambda row: (row["p50"], -row[quality]))
    eligible = [row for row in ordered if row["reference_label"] is None]
    front = []
    best_quality = float("-inf")
    for row in eligible:
        if row[quality] > best_quality:
            front.append(row)
            best_quality = row[quality]

    width, height, left, right, top, bottom = 820, 500, 85, 35, 50, 70
    plot_width, plot_height = width - left - right, height - top - bottom
    max_latency = max(row["p50"] for row in ordered) * 1.1
    minimum_quality = min(row[quality] for row in ordered)
    y_min = max(0.0, minimum_quality - max(0.01, (1 - minimum_quality) * 0.25))

    def point(row: dict) -> tuple[float, float]:
        return (
            left + row["p50"] / max_latency * plot_width,
            top + (1 - row[quality]) / (1 - y_min) * plot_height,
        )

    body = [
        f'<text class="axis" x="{left + plot_width / 2}" y="{height - 18}" text-anchor="middle">mean p50 inference latency (ms; lower is faster)</text>',
        f'<text class="axis" transform="translate(20 {top + plot_height / 2}) rotate(-90)" text-anchor="middle">mean {escape(quality_label)} (higher is better)</text>',
    ]
    legend_items = (("webgpu", "WebGPU"), ("wasm", "WASM"), ("native", "Native"))
    for index, (kind, label) in enumerate(legend_items):
        legend_x = left + index * 110
        body.extend(
            (
                f'<circle cx="{legend_x}" cy="24" r="6" fill="{PARETO_TYPE_COLORS[kind]}"/>',
                f'<text class="axis" x="{legend_x + 11}" y="28">{label}</text>',
            )
        )
    references = [row for row in ordered if row["reference_label"] is not None]
    assert len(references) == 1
    reference = references[0]
    reference_x = left + len(legend_items) * 110
    body.extend(
        (
            f'<circle cx="{reference_x}" cy="24" r="6" fill="{REFERENCE_COLOR}"/>',
            f'<text class="axis" x="{reference_x + 11}" y="28">{escape(reference["reference_label"])}</text>',
        )
    )
    for index in range(6):
        latency = max_latency * index / 5
        x = left + plot_width * index / 5
        quality_tick = y_min + (1 - y_min) * index / 5
        y = top + plot_height - plot_height * index / 5
        body.extend(
            (
                f'<line class="grid" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_height}"/>',
                f'<text class="axis" x="{x:.1f}" y="{top + plot_height + 22}" text-anchor="middle">{latency:.0f}</text>',
                f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}"/>',
                f'<text class="axis" x="{left - 10}" y="{y + 4:.1f}" text-anchor="end">{quality_tick:.3f}</text>',
            )
        )
    if len(front) > 1:
        body.append(
            f'<polyline fill="none" stroke="{FRONT_COLOR}" stroke-width="3" stroke-dasharray="7 5" points="'
            + " ".join(f"{point(row)[0]:.1f},{point(row)[1]:.1f}" for row in front)
            + '"/>'
        )
    points = [point(row) for row in ordered]
    label_boxes: list[tuple[float, float, float, float]] = []
    for row in ordered:
        x, y = point(row)
        on_front = row in front
        excluded = row["reference_label"] is not None
        label = row["reference_label"] if excluded else row["label"]
        body.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="8" fill="{REFERENCE_COLOR if excluded else PARETO_TYPE_COLORS[row["kind"]]}" stroke="{FRONT_COLOR if on_front else "#fff"}" stroke-width="3">'
            f'<title>{escape(row["label"])}: p50 {row["p50"]:.2f} ms, {escape(quality_label)} {row[quality]:.4f}{"; reference excluded from frontier" if excluded else ""}</title></circle>'
        )
        text_width = len(label) * 7.2
        candidates = (
            (12, -12, "start"),
            (12, 24, "start"),
            (-12, -12, "end"),
            (-12, 24, "end"),
            (12, -32, "start"),
            (12, 44, "start"),
            (-12, -32, "end"),
            (-12, 44, "end"),
        )
        selected: tuple[float, float, str, tuple[float, float, float, float]] | None = None
        for dx, dy, anchor in candidates:
            text_x, text_y = x + dx, y + dy
            x0 = text_x if anchor == "start" else text_x - text_width
            box = (x0, text_y - 15, x0 + text_width, text_y + 4)
            inside = box[0] >= 5 and box[2] <= width - 5 and box[1] >= top and box[3] <= top + plot_height
            overlaps_label = any(
                box[0] < other[2] and box[2] > other[0] and box[1] < other[3] and box[3] > other[1]
                for other in label_boxes
            )
            covers_point = any(
                box[0] - 6 < point_x < box[2] + 6 and box[1] - 6 < point_y < box[3] + 6
                for point_x, point_y in points
            )
            if inside and not overlaps_label and not covers_point:
                selected = (text_x, text_y, anchor, box)
                break
        assert selected is not None, f"No non-overlapping label position for {label}"
        text_x, text_y, anchor, box = selected
        label_boxes.append(box)
        body.append(
            f'<text class="label" x="{text_x:.1f}" y="{text_y:.1f}" text-anchor="{anchor}">{escape(label)}</text>'
        )
    return document(title, width, height, body)


def main() -> None:
    args = parse_args()
    results = json.loads(args.input.read_text(encoding="utf-8"))
    assert results["schemaVersion"] == 2
    task = results["task"]
    browser = browser_rows(results)
    all_rows = browser + native_rows(results)
    assert browser
    assert all_rows
    references = [row for row in all_rows if row["reference_label"] is not None]
    assert len(references) == 1
    reference = references[0]
    reference_label = reference["reference_label"]
    comparison_rows = [row for row in all_rows if row is not reference]
    latency_maximum = max(row["p95"] for row in all_rows) * 1.1
    args.output.mkdir(parents=True, exist_ok=True)
    plots = {
        "latency-percentiles.svg": horizontal_plot(
            "Inference latency averaged across benchmark datasets (ms)",
            "Inference latency (ms)",
            comparison_rows,
            (("p50", "p50"), ("p95", "p95")),
            0,
            latency_maximum,
            ((f"{reference_label} p50", reference["p50"]),),
        ),
    }
    if task == "classification":
        plots.update(
            {
                "accuracy.svg": horizontal_plot(
                    "Accuracy averaged across benchmark datasets",
                    "Accuracy",
                    browser,
                    (("accuracy", "accuracy"),),
                    max(0.0, min(row["accuracy"] for row in browser) - 0.05),
                    1.0,
                    ((f"{reference_label} accuracy", reference["accuracy"]),),
                ),
                "mcc.svg": horizontal_plot(
                    "MCC averaged across benchmark datasets",
                    "Matthews correlation coefficient",
                    browser,
                    (("mcc", "MCC"),),
                    max(0.0, min(row["mcc"] for row in browser) - 0.05),
                    1.0,
                    ((f"{reference_label} MCC", reference["mcc"]),),
                ),
                "pareto-speed-vs-accuracy.svg": pareto_plot(
                    all_rows,
                    "accuracy",
                    "accuracy",
                    "Inference speed vs accuracy",
                ),
            }
        )
    else:
        plots.update(
            {
                "r2.svg": horizontal_plot(
                    "R² averaged across regression benchmark datasets",
                    "R²",
                    comparison_rows,
                    (("r2", "R²"),),
                    max(-1.0, min(row["r2"] for row in all_rows) - 0.05),
                    min(1.0, max(row["r2"] for row in all_rows) + 0.05),
                    ((f"{reference_label} R²", reference["r2"]),),
                ),
                "errors.svg": horizontal_plot(
                    "Regression errors averaged across benchmark datasets",
                    "Target units (lower is better)",
                    comparison_rows,
                    (("mae", "MAE"), ("rmse", "RMSE")),
                    0,
                    max(max(row["mae"], row["rmse"]) for row in all_rows) * 1.1,
                    (
                        (f"{reference_label} MAE", reference["mae"]),
                        (f"{reference_label} RMSE", reference["rmse"]),
                    ),
                ),
                "pareto-speed-vs-r2.svg": pareto_plot(
                    all_rows,
                    "r2",
                    "R²",
                    "Inference speed vs regression quality",
                ),
            }
        )
    for name, svg in plots.items():
        (args.output / name).write_text(svg, encoding="utf-8")
    print(f"Wrote {len(plots)} SVG plots to {args.output}")


if __name__ == "__main__":
    main()
