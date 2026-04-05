#!/usr/bin/env python
"""Aggregate task-run perplexity deltas into CSV and Markdown tables."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate PLoRA task perplexity tables")
    p.add_argument(
        "--task-run",
        action="append",
        required=True,
        help="Mapping of task name to run dir, e.g. summarization=C:/runs/summarization",
    )
    p.add_argument("--output-dir", type=str, required=True)
    return p.parse_args()


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_task_mapping(raw: str) -> Tuple[str, Path]:
    if "=" not in raw:
        raise ValueError(f"Expected --task-run task=path, got: {raw}")
    task, path = raw.split("=", 1)
    return task.strip(), Path(path.strip())


def format_float(value: Any) -> str:
    return f"{float(value):.4f}"


def build_markdown(grouped: Dict[str, List[Dict[str, Any]]]) -> str:
    sections: List[str] = ["# Task Perplexity Change Tables", ""]
    for task in sorted(grouped):
        sections.append(f"## {task.title()}")
        sections.append("")
        sections.append("| Language | Code | Base PPL | Adapter PPL | Delta | Delta % | Direction |")
        sections.append("|---|---:|---:|---:|---:|---:|---|")
        for row in sorted(grouped[task], key=lambda item: item["language"]):
            sections.append(
                "| {language} | {lang} | {base} | {adapter} | {delta} | {delta_pct} | {direction} |".format(
                    language=row["language"],
                    lang=row["lang"],
                    base=format_float(row["base_ppl"]),
                    adapter=format_float(row["own_adapter_ppl"]),
                    delta=format_float(row["ppl_change"]),
                    delta_pct=format_float(row["ppl_change_pct"]),
                    direction=row["direction"],
                )
            )
        sections.append("")
    return "\n".join(sections).strip() + "\n"


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    combined_rows: List[Dict[str, Any]] = []
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    task_runs = [parse_task_mapping(raw) for raw in args.task_run]

    for declared_task, run_dir in task_runs:
        eval_csv = run_dir / "final_task_eval.csv"
        if not eval_csv.exists():
            raise FileNotFoundError(f"Missing final_task_eval.csv under {run_dir}")
        rows = read_csv_rows(eval_csv)
        for row in rows:
            task_name = row.get("task") or declared_task
            parsed = {
                "task": task_name,
                "lang": row["lang"],
                "language": row["language"],
                "base_ppl": float(row["base_ppl"]),
                "own_adapter_ppl": float(row["own_adapter_ppl"]),
                "ppl_change": float(row["ppl_change"]),
                "ppl_change_pct": float(row["ppl_change_pct"]),
                "direction": row["direction"],
                "run_dir": str(run_dir),
            }
            combined_rows.append(parsed)
            grouped[task_name].append(parsed)

    combined_rows.sort(key=lambda row: (row["task"], row["language"]))
    write_csv(
        out_dir / "perplexity_change_summary.csv",
        combined_rows,
        [
            "task",
            "lang",
            "language",
            "base_ppl",
            "own_adapter_ppl",
            "ppl_change",
            "ppl_change_pct",
            "direction",
            "run_dir",
        ],
    )

    language_codes = sorted({row["lang"] for row in combined_rows})
    tasks = sorted(grouped)
    wide_rows: List[Dict[str, Any]] = []
    lookup = {(row["lang"], row["task"]): row for row in combined_rows}
    for lang in language_codes:
        language_name = next(row["language"] for row in combined_rows if row["lang"] == lang)
        wide = {"lang": lang, "language": language_name}
        for task in tasks:
            row = lookup.get((lang, task))
            wide[f"{task}_base_ppl"] = row["base_ppl"] if row else ""
            wide[f"{task}_adapter_ppl"] = row["own_adapter_ppl"] if row else ""
            wide[f"{task}_ppl_change"] = row["ppl_change"] if row else ""
            wide[f"{task}_ppl_change_pct"] = row["ppl_change_pct"] if row else ""
            wide[f"{task}_direction"] = row["direction"] if row else ""
        wide_rows.append(wide)

    wide_fields = ["lang", "language"]
    for task in tasks:
        wide_fields.extend(
            [
                f"{task}_base_ppl",
                f"{task}_adapter_ppl",
                f"{task}_ppl_change",
                f"{task}_ppl_change_pct",
                f"{task}_direction",
            ]
        )
    write_csv(out_dir / "perplexity_change_wide.csv", wide_rows, wide_fields)
    (out_dir / "perplexity_change_tables.md").write_text(build_markdown(grouped), encoding="utf-8")


if __name__ == "__main__":
    main()

