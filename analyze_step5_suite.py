#!/usr/bin/env python
"""Analyze Step 5 experiment outputs and generate aggregate tables/figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze Step 5 suite outputs")
    p.add_argument(
        "--output-root",
        type=str,
        default=r"C:\Work\PLoRA\outputs\step5_suite_comprehensive",
        help="Root directory containing per-run Step 5 outputs",
    )
    p.add_argument(
        "--analysis-dir",
        type=str,
        default="",
        help="Directory to write aggregate tables and figures. Defaults to <output-root>/analysis",
    )
    return p.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def maybe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    return pd.read_csv(path)


def collect_runs(output_root: Path) -> Dict[str, pd.DataFrame]:
    run_rows: List[Dict[str, Any]] = []
    diag_frames: List[pd.DataFrame] = []
    loss_frames: List[pd.DataFrame] = []
    budget_frames: List[pd.DataFrame] = []
    cross_rows: List[pd.DataFrame] = []

    for run_dir in sorted(p for p in output_root.iterdir() if p.is_dir()):
        summary_path = run_dir / "run_summary.json"
        if not summary_path.exists():
            continue

        summary = load_json(summary_path)
        row = {
            "run_name": run_dir.name,
            "run_dir": str(run_dir),
            "budget_key": summary.get("budget_key"),
            "dataset_kind": summary.get("dataset_kind"),
            "lambda_chan": summary.get("lambda_chan"),
            "steps": summary.get("steps"),
            "model_id": summary.get("model_id"),
            "languages": ",".join(summary.get("languages", [])),
        }
        train_counts = summary.get("train_counts", {})
        eval_counts = summary.get("eval_counts", {})
        if train_counts:
            vals = list(train_counts.values())
            row["train_count_min"] = min(vals)
            row["train_count_max"] = max(vals)
            row["train_count_mean"] = sum(vals) / len(vals)
        if eval_counts:
            vals = list(eval_counts.values())
            row["eval_count_min"] = min(vals)
            row["eval_count_max"] = max(vals)
            row["eval_count_mean"] = sum(vals) / len(vals)
        final_diag = summary.get("final_diag_eval", {})
        if final_diag:
            vals = list(final_diag.values())
            row["mean_own_adapter_ppl"] = sum(vals) / len(vals)
            row["max_own_adapter_ppl"] = max(vals)
            row["min_own_adapter_ppl"] = min(vals)
        run_rows.append(row)

        diag_df = maybe_read_csv(run_dir / "final_diagonal_eval.csv")
        if diag_df is not None:
            diag_df["run_name"] = run_dir.name
            diag_frames.append(diag_df)

        loss_df = maybe_read_csv(run_dir / "train_loss_summary.csv")
        if loss_df is not None:
            loss_df["run_name"] = run_dir.name
            loss_frames.append(loss_df)

        budget_df = maybe_read_csv(run_dir / "language_budget_summary.csv")
        if budget_df is not None:
            budget_df["run_name"] = run_dir.name
            budget_frames.append(budget_df)

        cross_df = maybe_read_csv(run_dir / "final_cross_eval_matrix.csv")
        if cross_df is not None:
            cross_df["run_name"] = run_dir.name
            cross_rows.append(cross_df)

    out = {
        "runs": pd.DataFrame(run_rows),
        "diag": pd.concat(diag_frames, ignore_index=True) if diag_frames else pd.DataFrame(),
        "loss": pd.concat(loss_frames, ignore_index=True) if loss_frames else pd.DataFrame(),
        "budget": pd.concat(budget_frames, ignore_index=True) if budget_frames else pd.DataFrame(),
        "cross": pd.concat(cross_rows, ignore_index=True) if cross_rows else pd.DataFrame(),
    }
    return out


def build_config_summary(runs: pd.DataFrame, diag: pd.DataFrame, loss: pd.DataFrame) -> pd.DataFrame:
    if runs.empty or diag.empty:
        return pd.DataFrame()

    meta_cols = ["run_name", "budget_key", "lambda_chan", "train_count_mean"]
    meta = runs[meta_cols].copy()

    diag_merge = diag.merge(meta, on="run_name", how="left")
    diag_summary = (
        diag_merge.groupby(["budget_key", "lambda_chan", "train_count_mean", "lang"], as_index=False)
        .agg(
            n_runs=("own_adapter_ppl", "count"),
            mean_own_adapter_ppl=("own_adapter_ppl", "mean"),
            std_own_adapter_ppl=("own_adapter_ppl", "std"),
            mean_own_minus_best=("own_minus_best", "mean"),
        )
    )

    loss_summary = pd.DataFrame()
    if not loss.empty:
        loss_merge = loss.merge(meta, on="run_name", how="left")
        loss_summary = (
            loss_merge.groupby(["budget_key", "lambda_chan", "train_count_mean", "lang"], as_index=False)
            .agg(
                mean_final_logged_loss=("final_logged_loss", "mean"),
                mean_improvement_pct=("improvement_pct", "mean"),
            )
        )
        diag_summary = diag_summary.merge(
            loss_summary,
            on=["budget_key", "lambda_chan", "train_count_mean", "lang"],
            how="left",
        )

    return diag_summary.sort_values(["train_count_mean", "budget_key", "lambda_chan", "lang"])


def build_best_tables(runs: pd.DataFrame, config_summary: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    result: Dict[str, pd.DataFrame] = {}
    if runs.empty:
        result["best_runs"] = pd.DataFrame()
        result["best_configs"] = pd.DataFrame()
        result["best_by_language"] = pd.DataFrame()
        return result

    best_runs = runs.sort_values(["mean_own_adapter_ppl", "max_own_adapter_ppl"]).copy()
    result["best_runs"] = best_runs

    if config_summary.empty:
        result["best_configs"] = pd.DataFrame()
        result["best_by_language"] = pd.DataFrame()
        return result

    cfg = (
        config_summary.groupby(["budget_key", "lambda_chan", "train_count_mean"], as_index=False)
        .agg(
            mean_own_adapter_ppl=("mean_own_adapter_ppl", "mean"),
            mean_own_minus_best=("mean_own_minus_best", "mean"),
            mean_final_logged_loss=("mean_final_logged_loss", "mean"),
            mean_improvement_pct=("mean_improvement_pct", "mean"),
        )
        .sort_values(["mean_own_adapter_ppl", "mean_final_logged_loss"])
    )
    result["best_configs"] = cfg

    best_by_lang = (
        config_summary.sort_values(["lang", "mean_own_adapter_ppl", "mean_final_logged_loss"])
        .groupby("lang", as_index=False)
        .first()
    )
    result["best_by_language"] = best_by_lang
    return result


def save_tables(analysis_dir: Path, tables: Dict[str, pd.DataFrame]) -> None:
    for name, df in tables.items():
        if df.empty:
            continue
        df.to_csv(analysis_dir / f"{name}.csv", index=False)


def plot_overall_scaling(analysis_dir: Path, cfg: pd.DataFrame) -> None:
    if cfg.empty:
        return
    overall = (
        cfg.groupby(["budget_key", "lambda_chan", "train_count_mean"], as_index=False)
        .agg(mean_own_adapter_ppl=("mean_own_adapter_ppl", "mean"))
        .sort_values("train_count_mean")
    )
    plt.figure(figsize=(10, 6))
    for (budget_key, lambda_chan), g in overall.groupby(["budget_key", "lambda_chan"]):
        plt.plot(
            g["train_count_mean"],
            g["mean_own_adapter_ppl"],
            marker="o",
            linewidth=2,
            label=f"{budget_key}, lambda={lambda_chan:g}",
        )
    plt.xlabel("Train Examples Per Language")
    plt.ylabel("Mean Own-Adapter PPL")
    plt.title("Step 5 Scaling Curve Across Configurations")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(analysis_dir / "overall_scaling_curve.png", dpi=180)
    plt.close()


def plot_language_scaling(analysis_dir: Path, cfg: pd.DataFrame) -> None:
    if cfg.empty:
        return
    langs = sorted(cfg["lang"].unique())
    ncols = 3
    nrows = (len(langs) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows), squeeze=False)
    for ax, lang in zip(axes.ravel(), langs):
        sub = cfg[cfg["lang"] == lang].sort_values("train_count_mean")
        for (budget_key, lambda_chan), g in sub.groupby(["budget_key", "lambda_chan"]):
            ax.plot(
                g["train_count_mean"],
                g["mean_own_adapter_ppl"],
                marker="o",
                linewidth=1.6,
                label=f"{budget_key}, λ={lambda_chan:g}",
            )
        ax.set_title(lang)
        ax.set_xlabel("Train Count")
        ax.set_ylabel("Mean Own PPL")
        ax.grid(alpha=0.25)
    for ax in axes.ravel()[len(langs):]:
        ax.axis("off")
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=8)
    fig.suptitle("Per-Language Scaling Curves", y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(analysis_dir / "per_language_scaling_curves.png", dpi=180)
    plt.close(fig)


def plot_config_bars(analysis_dir: Path, best_configs: pd.DataFrame) -> None:
    if best_configs.empty:
        return
    top = best_configs.head(12).copy()
    top["label"] = top.apply(
        lambda r: f"{r['budget_key'].replace('_budget', '')} | λ={r['lambda_chan']:g} | N={int(r['train_count_mean'])}",
        axis=1,
    )
    plt.figure(figsize=(11, 6))
    plt.barh(top["label"][::-1], top["mean_own_adapter_ppl"][::-1])
    plt.xlabel("Mean Own-Adapter PPL")
    plt.title("Top Configurations")
    plt.tight_layout()
    plt.savefig(analysis_dir / "top_configurations.png", dpi=180)
    plt.close()


def plot_improvement_heatmap(analysis_dir: Path, cfg: pd.DataFrame) -> None:
    if cfg.empty or "mean_improvement_pct" not in cfg.columns:
        return
    overall = (
        cfg.groupby(["budget_key", "lambda_chan", "train_count_mean"], as_index=False)
        .agg(mean_improvement_pct=("mean_improvement_pct", "mean"))
    )
    pivot = overall.pivot_table(
        index=["budget_key", "lambda_chan"],
        columns="train_count_mean",
        values="mean_improvement_pct",
    )
    plt.figure(figsize=(8, 5))
    plt.imshow(pivot.values, aspect="auto")
    plt.xticks(range(len(pivot.columns)), [str(int(x)) for x in pivot.columns])
    plt.yticks(
        range(len(pivot.index)),
        [f"{idx[0].replace('_budget', '')}, λ={idx[1]:g}" for idx in pivot.index],
    )
    plt.colorbar(label="Mean Improvement %")
    plt.xlabel("Train Count")
    plt.title("Loss Improvement Heatmap")
    plt.tight_layout()
    plt.savefig(analysis_dir / "loss_improvement_heatmap.png", dpi=180)
    plt.close()


def plot_budget_summary(analysis_dir: Path, budget: pd.DataFrame) -> None:
    if budget.empty:
        return
    base = budget.sort_values("rank_sum").drop_duplicates("lang")
    plt.figure(figsize=(9, 5))
    plt.barh(base["language"], base["rank_sum"])
    plt.xlabel("Total Routed Rank")
    plt.title("Language Budget Summary")
    plt.tight_layout()
    plt.savefig(analysis_dir / "language_budget_summary.png", dpi=180)
    plt.close()


def plot_best_cross_eval(analysis_dir: Path, cross: pd.DataFrame, best_run_name: str) -> None:
    if cross.empty:
        return
    sub = cross[cross["run_name"] == best_run_name].copy()
    if sub.empty:
        return
    adapter_cols = [c for c in sub.columns if c not in {"eval_lang", "eval_language", "run_name"}]
    matrix = sub[adapter_cols].to_numpy(dtype=float)
    plt.figure(figsize=(9, 7))
    plt.imshow(matrix, aspect="auto")
    plt.xticks(range(len(adapter_cols)), adapter_cols, rotation=45, ha="right")
    plt.yticks(range(len(sub)), sub["eval_language"])
    plt.colorbar(label="PPL")
    plt.title(f"Best Run Cross-Eval Matrix: {best_run_name}")
    plt.tight_layout()
    plt.savefig(analysis_dir / "best_run_cross_eval_heatmap.png", dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    analysis_dir = Path(args.analysis_dir) if args.analysis_dir else output_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    collected = collect_runs(output_root)
    cfg_summary = build_config_summary(collected["runs"], collected["diag"], collected["loss"])
    best_tables = build_best_tables(collected["runs"], cfg_summary)

    save_tables(
        analysis_dir,
        {
            "run_table": collected["runs"],
            "diagonal_eval_table": collected["diag"],
            "loss_summary_table": collected["loss"],
            "budget_summary_table": collected["budget"],
            "config_summary": cfg_summary,
            "best_runs": best_tables["best_runs"],
            "best_configs": best_tables["best_configs"],
            "best_by_language": best_tables["best_by_language"],
        },
    )

    plot_overall_scaling(analysis_dir, cfg_summary)
    plot_language_scaling(analysis_dir, cfg_summary)
    plot_config_bars(analysis_dir, best_tables["best_configs"])
    plot_improvement_heatmap(analysis_dir, cfg_summary)
    plot_budget_summary(analysis_dir, collected["budget"])
    if not best_tables["best_runs"].empty:
        plot_best_cross_eval(
            analysis_dir,
            collected["cross"],
            best_tables["best_runs"].iloc[0]["run_name"],
        )

    summary = {
        "output_root": str(output_root),
        "analysis_dir": str(analysis_dir),
        "n_completed_runs": int(len(collected["runs"])),
        "n_diagonal_eval_rows": int(len(collected["diag"])),
        "n_loss_rows": int(len(collected["loss"])),
        "n_cross_eval_rows": int(len(collected["cross"])),
    }
    (analysis_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
