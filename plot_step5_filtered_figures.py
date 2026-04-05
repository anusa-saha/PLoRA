#!/usr/bin/env python
"""Generate Step 5 figures while excluding selected languages.

Expected inputs come from a Step 5 run directory, for example:
  - train_log.jsonl
  - eval_history.csv
  - final_diagonal_eval.csv
  - final_cross_eval_matrix.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd


LANGUAGE_NAMES = {
    "ben_Beng": "Bengali",
    "cmn_Hans": "Chinese",
    "eng_Latn": "English",
    "fra_Latn": "French",
    "hin_Deva": "Hindi",
    "mar_Deva": "Marathi",
    "nld_Latn": "Dutch",
    "pol_Latn": "Polish",
    "urd_Arab": "Urdu",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Step 5 figures excluding selected languages")
    p.add_argument("--run-dir", type=str, required=True, help="Step 5 run directory")
    p.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Directory for filtered figures. Defaults to <run-dir>/filtered_figures",
    )
    p.add_argument(
        "--exclude-languages",
        type=str,
        default="ben_Beng,urd_Arab",
        help="Comma-separated language codes to exclude",
    )
    p.add_argument("--smooth-window", type=int, default=25)
    return p.parse_args()


def parse_excluded(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def load_train_log(path: Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return pd.DataFrame(rows)


def smooth_series(values: List[float], window: int) -> List[float]:
    out: List[float] = []
    for idx in range(len(values)):
        lo = max(0, idx - window + 1)
        chunk = values[lo : idx + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def maybe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def plot_train_loss(run_dir: Path, output_dir: Path, excluded: List[str], smooth_window: int) -> None:
    path = run_dir / "train_log.jsonl"
    if not path.exists():
        return
    df = load_train_log(path)
    if df.empty:
        return
    df = df[~df["lang"].isin(excluded)].copy()
    if df.empty:
        return

    plt.figure(figsize=(12, 7))
    for lang, group in sorted(df.groupby("lang"), key=lambda kv: kv[0]):
        group = group.sort_values("step")
        xs = group["step"].tolist()
        ys = group["loss"].tolist()
        plt.plot(xs, smooth_series(ys, smooth_window), linewidth=1.7, label=LANGUAGE_NAMES.get(lang, lang))
    plt.xlabel("Global Step")
    plt.ylabel("Smoothed Train Loss")
    plt.title("Step 5 Train Loss by Language (Filtered)")
    plt.grid(alpha=0.25)
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "train_loss_by_language_filtered.png", dpi=180)
    plt.close()


def plot_eval_over_time(run_dir: Path, output_dir: Path, excluded: List[str]) -> None:
    path = run_dir / "eval_history.csv"
    df = maybe_read_csv(path)
    if df.empty:
        return

    plt.figure(figsize=(12, 7))
    for col in sorted(c for c in df.columns if c.startswith("ppl_")):
        lang = col.replace("ppl_", "", 1)
        if lang in excluded:
            continue
        plt.plot(df["step"], df[col], marker="o", linewidth=1.6, label=LANGUAGE_NAMES.get(lang, lang))
    plt.xlabel("Global Step")
    plt.ylabel("Perplexity")
    plt.title("Step 5 Diagonal Eval PPL (Filtered)")
    plt.grid(alpha=0.25)
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "eval_ppl_over_time_filtered.png", dpi=180)
    plt.close()


def plot_final_diagonal(run_dir: Path, output_dir: Path, excluded: List[str]) -> None:
    path = run_dir / "final_diagonal_eval.csv"
    df = maybe_read_csv(path)
    if df.empty:
        return
    df = df[~df["lang"].isin(excluded)].copy()
    if df.empty:
        return
    df = df.sort_values("own_adapter_ppl", ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(df["language"], df["own_adapter_ppl"])
    plt.xlabel("Final Own-Adapter PPL")
    plt.title("Final Diagonal Eval (Filtered)")
    plt.tight_layout()
    plt.savefig(output_dir / "final_diagonal_eval_filtered.png", dpi=180)
    plt.close()


def plot_cross_eval(run_dir: Path, output_dir: Path, excluded: List[str]) -> None:
    path = run_dir / "final_cross_eval_matrix.csv"
    df = maybe_read_csv(path)
    if df.empty:
        return

    keep_rows = ~df["eval_lang"].isin(excluded)
    df = df[keep_rows].copy()
    adapter_cols = [c for c in df.columns if c not in {"eval_lang", "eval_language"} and c not in excluded]
    if df.empty or not adapter_cols:
        return

    matrix = df[adapter_cols].to_numpy(dtype=float)
    plt.figure(figsize=(9, 7))
    plt.imshow(matrix, aspect="auto")
    plt.xticks(range(len(adapter_cols)), [LANGUAGE_NAMES.get(c, c) for c in adapter_cols], rotation=45, ha="right")
    plt.yticks(range(len(df)), df["eval_language"])
    plt.colorbar(label="PPL")
    plt.title("Final Cross-Language PPL Matrix (Filtered)")
    plt.tight_layout()
    plt.savefig(output_dir / "cross_eval_heatmap_filtered.png", dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "filtered_figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    excluded = parse_excluded(args.exclude_languages)

    plot_train_loss(run_dir, output_dir, excluded, args.smooth_window)
    plot_eval_over_time(run_dir, output_dir, excluded)
    plot_final_diagonal(run_dir, output_dir, excluded)
    plot_cross_eval(run_dir, output_dir, excluded)

    summary = {
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "excluded_languages": excluded,
    }
    (output_dir / "filtered_figures_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
