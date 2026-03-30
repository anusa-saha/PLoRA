#!/usr/bin/env python
"""Step 5 experiment suite runner.

Builds and executes a paper-style ablation grid over the Step 5 training script.
The suite is intentionally focused on the knobs the current implementation supports:
budget allocation, regularization strength, training-set size, and random seed.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import logging
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List


LOGGER = logging.getLogger("plora.step5.suite")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a comprehensive Step 5 experiment suite")
    p.add_argument("--python-exe", type=str, default=sys.executable)
    p.add_argument("--train-script", type=str, default="plora_step5_language_routed_training.py")
    p.add_argument("--rank-json", type=str, default="plora_step4_rank_budgets.json")
    p.add_argument("--output-root", type=str, default="outputs/step5_suite_comprehensive")
    p.add_argument("--suite-name", type=str, default="flores2k_paper_grid")
    p.add_argument("--model-id", type=str, default="")
    p.add_argument("--dataset-kind", type=str, default="flores200_mirror")
    p.add_argument("--budget-keys", type=str, default="fair_budget,equal_budget")
    p.add_argument("--lambda-values", type=str, default="0.0,0.0005,0.001")
    p.add_argument("--sample-sizes", type=str, default="250,500,1000,2000")
    p.add_argument("--seeds", type=str, default="11,22,33")
    p.add_argument("--max-eval-samples", type=int, default=9)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--train-batch-size", type=int, default=1)
    p.add_argument("--eval-batch-size", type=int, default=1)
    p.add_argument("--grad-accum-steps", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--target-languages", type=str, default="")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--use-bf16", action="store_true")
    p.add_argument("--skip-cross-eval", action="store_true")
    p.add_argument("--skip-report-plots", action="store_true")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--max-runs", type=int, default=0, help="Optional cap for partial launches.")
    p.add_argument("--print-only", action="store_true", help="Only materialize the manifest.")
    return p.parse_args()


def parse_csv_list(raw: str, cast) -> List[Any]:
    return [cast(x.strip()) for x in raw.split(",") if x.strip()]


def slugify_float(value: float) -> str:
    if math.isclose(value, round(value)):
        return str(int(round(value)))
    return str(value).replace("-", "m").replace(".", "p")


def build_experiment_name(cfg: Dict[str, Any]) -> str:
    parts = [
        f"budget-{cfg['budget_key'].replace('_budget', '')}",
        f"lambda-{slugify_float(cfg['lambda_chan'])}",
        f"n-{cfg['max_train_samples']}",
        f"seed-{cfg['seed']}",
    ]
    return "__".join(parts)


def build_command(args: argparse.Namespace, cfg: Dict[str, Any], run_dir: Path) -> List[str]:
    cmd = [
        args.python_exe,
        str(Path(args.train_script)),
        "--rank-json",
        args.rank_json,
        "--output-dir",
        str(run_dir),
        "--budget-key",
        cfg["budget_key"],
        "--dataset-kind",
        args.dataset_kind,
        "--max-train-samples",
        str(cfg["max_train_samples"]),
        "--max-eval-samples",
        str(args.max_eval_samples),
        "--max-steps",
        str(args.max_steps),
        "--train-batch-size",
        str(args.train_batch_size),
        "--eval-batch-size",
        str(args.eval_batch_size),
        "--grad-accum-steps",
        str(args.grad_accum_steps),
        "--learning-rate",
        str(args.learning_rate),
        "--weight-decay",
        str(args.weight_decay),
        "--max-length",
        str(args.max_length),
        "--seed",
        str(cfg["seed"]),
        "--log-every",
        str(args.log_every),
        "--eval-every",
        str(args.eval_every),
        "--save-every",
        str(args.save_every),
        "--lambda-chan",
        str(cfg["lambda_chan"]),
    ]
    if args.model_id:
        cmd.extend(["--model-id", args.model_id])
    if args.target_languages:
        cmd.extend(["--target-languages", args.target_languages])
    if args.trust_remote_code:
        cmd.append("--trust-remote-code")
    if args.use_bf16:
        cmd.append("--use-bf16")
    if args.skip_cross_eval:
        cmd.append("--skip-cross-eval")
    if args.skip_report_plots:
        cmd.append("--skip-report-plots")
    return cmd


def write_json(path: Path, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def collect_suite_results(output_root: Path) -> Dict[str, List[Dict[str, Any]]]:
    run_rows: List[Dict[str, Any]] = []
    diag_rows: List[Dict[str, Any]] = []
    loss_rows: List[Dict[str, Any]] = []

    for run_dir in sorted(output_root.iterdir()):
        if not run_dir.is_dir():
            continue
        summary_path = run_dir / "run_summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            row = {"run_name": run_dir.name}
            row.update(summary)
            run_rows.append(row)

        diag_path = run_dir / "final_diagonal_eval.csv"
        if diag_path.exists():
            with open(diag_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for record in reader:
                    record["run_name"] = run_dir.name
                    diag_rows.append(record)

        loss_path = run_dir / "train_loss_summary.csv"
        if loss_path.exists():
            with open(loss_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for record in reader:
                    record["run_name"] = run_dir.name
                    loss_rows.append(record)

    return {
        "run_rows": run_rows,
        "diag_rows": diag_rows,
        "loss_rows": loss_rows,
    }


def build_seed_aggregate(run_rows: List[Dict[str, Any]], diag_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    run_meta: Dict[str, Dict[str, Any]] = {}
    for row in run_rows:
        run_meta[row["run_name"]] = row

    grouped: Dict[tuple, List[float]] = {}
    for row in diag_rows:
        run_name = row["run_name"]
        meta = run_meta.get(run_name)
        if not meta:
            continue
        key = (
            meta["budget_key"],
            float(meta["lambda_chan"]),
            int(meta["train_counts"][row["lang"]]),
            row["lang"],
        )
        grouped.setdefault(key, []).append(float(row["own_adapter_ppl"]))

    out: List[Dict[str, Any]] = []
    for key, vals in sorted(grouped.items()):
        budget_key, lambda_chan, train_count, lang = key
        mean = sum(vals) / len(vals)
        var = sum((x - mean) ** 2 for x in vals) / len(vals)
        out.append(
            {
                "budget_key": budget_key,
                "lambda_chan": lambda_chan,
                "train_count": train_count,
                "lang": lang,
                "n_seeds": len(vals),
                "mean_own_adapter_ppl": mean,
                "std_own_adapter_ppl": math.sqrt(var),
            }
        )
    return out


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    budget_keys = parse_csv_list(args.budget_keys, str)
    lambda_values = parse_csv_list(args.lambda_values, float)
    sample_sizes = parse_csv_list(args.sample_sizes, int)
    seeds = parse_csv_list(args.seeds, int)

    matrix: List[Dict[str, Any]] = []
    for budget_key, lambda_chan, max_train_samples, seed in itertools.product(
        budget_keys,
        lambda_values,
        sample_sizes,
        seeds,
    ):
        cfg = {
            "budget_key": budget_key,
            "lambda_chan": lambda_chan,
            "max_train_samples": max_train_samples,
            "seed": seed,
        }
        run_name = build_experiment_name(cfg)
        run_dir = output_root / run_name
        cmd = build_command(args, cfg, run_dir)
        matrix.append(
            {
                **cfg,
                "run_name": run_name,
                "run_dir": str(run_dir),
                "command": cmd,
            }
        )

    manifest = {
        "suite_name": args.suite_name,
        "dataset_kind": args.dataset_kind,
        "n_runs": len(matrix),
        "matrix": matrix,
    }
    write_json(output_root / "suite_manifest.json", manifest)

    if args.print_only:
        LOGGER.info("Wrote manifest for %d runs to %s", len(matrix), output_root / "suite_manifest.json")
        return

    completed = 0
    launched = 0
    for item in matrix:
        run_dir = Path(item["run_dir"])
        summary_path = run_dir / "run_summary.json"
        if args.skip_existing and summary_path.exists():
            LOGGER.info("Skipping existing run: %s", item["run_name"])
            completed += 1
            continue
        if args.max_runs and launched >= args.max_runs:
            break

        run_dir.mkdir(parents=True, exist_ok=True)
        write_json(run_dir / "suite_run_config.json", item)
        LOGGER.info("Launching [%d/%d] %s", launched + 1, len(matrix), item["run_name"])
        LOGGER.info("Command: %s", " ".join(item["command"]))
        subprocess.run(item["command"], check=True)
        launched += 1
        completed += 1

    collected = collect_suite_results(output_root)
    write_json(output_root / "suite_run_summaries.json", collected["run_rows"])

    if collected["run_rows"]:
        summary_fieldnames = sorted({k for row in collected["run_rows"] for k in row.keys()})
        write_csv(output_root / "suite_run_summaries.csv", collected["run_rows"], summary_fieldnames)
    if collected["diag_rows"]:
        diag_fieldnames = sorted({k for row in collected["diag_rows"] for k in row.keys()})
        write_csv(output_root / "suite_final_diagonal_eval.csv", collected["diag_rows"], diag_fieldnames)
    if collected["loss_rows"]:
        loss_fieldnames = sorted({k for row in collected["loss_rows"] for k in row.keys()})
        write_csv(output_root / "suite_train_loss_summary.csv", collected["loss_rows"], loss_fieldnames)

    seed_aggregate = build_seed_aggregate(collected["run_rows"], collected["diag_rows"])
    if seed_aggregate:
        write_csv(
            output_root / "suite_seed_aggregate.csv",
            seed_aggregate,
            [
                "budget_key",
                "lambda_chan",
                "train_count",
                "lang",
                "n_seeds",
                "mean_own_adapter_ppl",
                "std_own_adapter_ppl",
            ],
        )

    LOGGER.info("Suite finished. Completed=%d launched_this_invocation=%d", completed, launched)


if __name__ == "__main__":
    main()
