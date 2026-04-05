#!/usr/bin/env python
"""Run task-specific PLoRA channel budgeting, training, and reporting."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the full PLoRA task suite on multiple tasks")
    p.add_argument("--dataset-manifest", type=str, required=True)
    p.add_argument("--output-root", type=str, required=True)
    p.add_argument("--python-exe", type=str, default=sys.executable)
    p.add_argument("--tasks", type=str, default="summarization,qa,sentiment")
    p.add_argument("--model-id", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--budget-key", type=str, default="fair_budget", choices=["fair_budget", "equal_budget"])
    p.add_argument("--target-languages", type=str, default="")
    p.add_argument("--probe-samples", type=int, default=1000)
    p.add_argument("--channel-train-samples", type=int, default=512)
    p.add_argument("--max-train-samples", type=int, default=2000)
    p.add_argument("--max-eval-samples", type=int, default=256)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--train-batch-size", type=int, default=2)
    p.add_argument("--eval-batch-size", type=int, default=2)
    p.add_argument("--grad-accum-steps", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--max-length", type=int, default=768)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lambda-chan", type=float, default=0.0)
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--use-bf16", action="store_true")
    p.add_argument("--skip-cross-eval", action="store_true")
    p.add_argument("--skip-channel-budget", action="store_true")
    p.add_argument("--skip-training", action="store_true")
    p.add_argument("--print-only", action="store_true")
    return p.parse_args()


def run_command(cmd: List[str]) -> None:
    print(">>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    task_run_dirs: Dict[str, Path] = {}

    for task in tasks:
        task_root = output_root / task
        channel_dir = task_root / "channel_budget"
        train_dir = task_root / "training"
        rank_json = channel_dir / f"{task}_rank_budgets.json"

        if not args.skip_channel_budget:
            channel_cmd = [
                args.python_exe,
                "plora_task_channel_budget.py",
                "--dataset-manifest",
                args.dataset_manifest,
                "--task",
                task,
                "--model-id",
                args.model_id,
                "--output-dir",
                str(channel_dir),
                "--probe-samples",
                str(args.probe_samples),
                "--train-samples-for-budget",
                str(args.channel_train_samples),
                "--eval-samples",
                str(args.max_eval_samples),
                "--max-length",
                str(args.max_length),
                "--seed",
                str(args.seed),
            ]
            if args.target_languages:
                channel_cmd.extend(["--target-languages", args.target_languages])
            if args.trust_remote_code:
                channel_cmd.append("--trust-remote-code")
            if args.use_bf16:
                channel_cmd.append("--use-bf16")
            if args.print_only:
                print(">>>", " ".join(channel_cmd), flush=True)
            else:
                run_command(channel_cmd)

        if not args.skip_training:
            train_cmd = [
                args.python_exe,
                "plora_task_language_routed_training.py",
                "--dataset-manifest",
                args.dataset_manifest,
                "--task",
                task,
                "--rank-json",
                str(rank_json),
                "--budget-key",
                args.budget_key,
                "--model-id",
                args.model_id,
                "--output-dir",
                str(train_dir),
                "--max-train-samples",
                str(args.max_train_samples),
                "--max-eval-samples",
                str(args.max_eval_samples),
                "--train-batch-size",
                str(args.train_batch_size),
                "--eval-batch-size",
                str(args.eval_batch_size),
                "--grad-accum-steps",
                str(args.grad_accum_steps),
                "--max-steps",
                str(args.max_steps),
                "--learning-rate",
                str(args.learning_rate),
                "--weight-decay",
                str(args.weight_decay),
                "--max-length",
                str(args.max_length),
                "--seed",
                str(args.seed),
                "--lambda-chan",
                str(args.lambda_chan),
            ]
            if args.target_languages:
                train_cmd.extend(["--target-languages", args.target_languages])
            if args.trust_remote_code:
                train_cmd.append("--trust-remote-code")
            if args.use_bf16:
                train_cmd.append("--use-bf16")
            if args.skip_cross_eval:
                train_cmd.append("--skip-cross-eval")
            if args.print_only:
                print(">>>", " ".join(train_cmd), flush=True)
            else:
                run_command(train_cmd)
        task_run_dirs[task] = train_dir

    report_cmd = [
        args.python_exe,
        "plora_task_perplexity_report.py",
        "--output-dir",
        str(output_root / "report"),
    ]
    for task, run_dir in task_run_dirs.items():
        report_cmd.extend(["--task-run", f"{task}={run_dir}"])
    if args.print_only:
        print(">>>", " ".join(report_cmd), flush=True)
    else:
        run_command(report_cmd)


if __name__ == "__main__":
    main()

