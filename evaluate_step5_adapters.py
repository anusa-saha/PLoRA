#!/usr/bin/env python
"""Evaluate Step 5 adapters on a separate multilingual corpus and generate figures.

Default eval source is OPUS-100, which is distinct from the FLORES-based Step 5
training setup used in the current experiments.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer


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

LANG_TO_OPUS = {
    "ben_Beng": ("bn-en", "bn"),
    "cmn_Hans": ("en-zh", "zh"),
    "eng_Latn": ("en-fr", "en"),
    "fra_Latn": ("en-fr", "fr"),
    "hin_Deva": ("en-hi", "hi"),
    "mar_Deva": ("en-mr", "mr"),
    "nld_Latn": ("en-nl", "nl"),
    "pol_Latn": ("en-pl", "pl"),
    "urd_Arab": ("en-ur", "ur"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Step 5 adapters on separate-source multilingual data")
    p.add_argument("--run-dir", type=str, required=True, help="Step 5 output directory containing final/ and run_summary.json")
    p.add_argument("--output-dir", type=str, default="", help="Directory to write eval artifacts. Defaults to <run-dir>/separate_eval")
    p.add_argument("--model-id", type=str, default="", help="Override base model id")
    p.add_argument("--eval-samples", type=int, default=300)
    p.add_argument("--skip-samples", type=int, default=100)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--use-bf16", action="store_true")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--skip-cross-eval", action="store_true")
    return p.parse_args()


def write_json(path: Path, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_run_summary(run_dir: Path) -> Dict[str, Any]:
    path = run_dir / "run_summary.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing run summary: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def find_latest_checkpoint_dir(run_dir: Path) -> Path | None:
    checkpoints = []
    for p in run_dir.iterdir():
        if p.is_dir() and p.name.startswith("checkpoint_step_"):
            try:
                step = int(p.name.replace("checkpoint_step_", "", 1))
            except ValueError:
                continue
            checkpoints.append((step, p))
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: x[0], reverse=True)
    return checkpoints[0][1]


def resolve_adapter_dir(adapter_dir: Path) -> Path:
    if (adapter_dir / "adapter_config.json").exists():
        return adapter_dir
    child_dirs = [p for p in adapter_dir.iterdir() if p.is_dir()]
    for child in child_dirs:
        if (child / "adapter_config.json").exists():
            return child
    return adapter_dir


def adapter_root_is_valid(root: Path) -> bool:
    if not root.exists() or not root.is_dir():
        return False
    adapter_dirs = [p for p in root.iterdir() if p.is_dir() and p.name.startswith("adapter_")]
    if not adapter_dirs:
        return False
    return all((resolve_adapter_dir(p) / "adapter_config.json").exists() for p in adapter_dirs)


def resolve_artifact_root(run_dir: Path) -> Path:
    final_dir = run_dir / "final"
    if adapter_root_is_valid(final_dir):
        return final_dir
    latest_ckpt = find_latest_checkpoint_dir(run_dir)
    if latest_ckpt and adapter_root_is_valid(latest_ckpt):
        return latest_ckpt
    raise FileNotFoundError(
        "Could not find a valid adapter artifact root. Checked "
        f"'{final_dir}' and latest checkpoint under '{run_dir}'."
    )


def build_eval_texts(
    tokenizer,
    lang_code: str,
    eval_samples: int,
    skip_samples: int,
    max_length: int,
) -> List[str]:
    cfg, field = LANG_TO_OPUS[lang_code]
    ds = load_dataset("Helsinki-NLP/opus-100", cfg, split="train", streaming=True)
    texts: List[str] = []
    seen_valid = 0
    for row in ds:
        translation = row.get("translation", {})
        text = translation.get(field, "")
        if not isinstance(text, str) or not text.strip():
            continue
        token_ids = tokenizer(text, truncation=True, max_length=max_length, padding=False)["input_ids"]
        if len(token_ids) <= 8:
            continue
        seen_valid += 1
        if seen_valid <= skip_samples:
            continue
        texts.append(text.strip())
        if len(texts) >= eval_samples:
            break
    if len(texts) < eval_samples:
        raise RuntimeError(f"Only found {len(texts)} eval texts for {lang_code}; need {eval_samples}")
    return texts


def tokenize_eval_texts(tokenizer, texts: List[str], max_length: int) -> Dataset:
    raw = Dataset.from_dict({"text": texts})

    def _tok(batch):
        tok = tokenizer(batch["text"], truncation=True, max_length=max_length, padding=False)
        tok["labels"] = [ids[:] for ids in tok["input_ids"]]
        return tok

    return raw.map(_tok, batched=True, remove_columns=["text"])


def make_causal_lm_collator(tokenizer):
    pad_id = tokenizer.pad_token_id

    def _collate(features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids = []
        attention_mask = []
        labels = []
        for f in features:
            ids = list(f["input_ids"])
            mask = list(f["attention_mask"])
            labs = list(f["labels"])
            pad_len = max_len - len(ids)
            input_ids.append(ids + [pad_id] * pad_len)
            attention_mask.append(mask + [0] * pad_len)
            labels.append(labs + [-100] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    return _collate


def make_loader(ds: Dataset, tokenizer, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=make_causal_lm_collator(tokenizer),
        drop_last=False,
        num_workers=num_workers,
    )


def evaluate_loader(model, loader: DataLoader, device: torch.device, adapter_name: str | None = None) -> float:
    model.eval()
    losses: List[float] = []
    if adapter_name is None:
        adapter_ctx = model.disable_adapter()
    else:
        model.set_adapter(adapter_name)
        adapter_ctx = nullcontext()
    with adapter_ctx:
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(**batch)
                losses.append(float(out.loss.detach().float().item()))
    return float(math.exp(sum(losses) / len(losses))) if losses else float("inf")


def load_model_and_adapters(
    run_dir: Path,
    model_id: str,
    trust_remote_code: bool,
    use_bf16: bool,
) -> Tuple[PeftModel, AutoTokenizer, torch.device, List[str]]:
    artifact_root = resolve_artifact_root(run_dir)
    adapter_dirs = sorted(p for p in artifact_root.iterdir() if p.is_dir() and p.name.startswith("adapter_"))
    if not adapter_dirs:
        raise FileNotFoundError(f"No adapter directories found under {artifact_root}")

    tokenizer_dir = artifact_root / "tokenizer"
    if not tokenizer_dir.exists():
        tokenizer_dir = run_dir / "final" / "tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if (use_bf16 and torch.cuda.is_available()) else torch.float16
    if not torch.cuda.is_available():
        dtype = torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if not torch.cuda.is_available():
        base.to(device)

    first = resolve_adapter_dir(adapter_dirs[0])
    first_name = first.name.replace("adapter_", "", 1)
    if first_name not in LANGUAGE_NAMES:
        first_name = adapter_dirs[0].name.replace("adapter_", "", 1)
    model = PeftModel.from_pretrained(base, str(first), adapter_name=first_name)
    adapter_names = [first_name]
    for adapter_dir in adapter_dirs[1:]:
        resolved = resolve_adapter_dir(adapter_dir)
        name = adapter_dir.name.replace("adapter_", "", 1)
        model.load_adapter(str(resolved), adapter_name=name)
        adapter_names.append(name)
    return model, tokenizer, device, adapter_names


def plot_eval_summary(
    output_dir: Path,
    summary_df: pd.DataFrame,
    cross_df: pd.DataFrame,
) -> None:
    plt.figure(figsize=(10, 6))
    sorted_df = summary_df.sort_values("adapted_ppl")
    plt.barh(sorted_df["language"], sorted_df["adapted_ppl"], label="Adapted")
    plt.scatter(sorted_df["base_ppl"], sorted_df["language"], color="black", label="Base", zorder=3)
    plt.xlabel("Perplexity")
    plt.title("Base vs Adapted PPL on Separate OPUS Eval")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "base_vs_adapted_ppl.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.barh(sorted_df["language"], sorted_df["improvement_pct"])
    plt.xlabel("Improvement over Base (%)")
    plt.title("Relative Improvement by Language")
    plt.tight_layout()
    plt.savefig(output_dir / "improvement_pct.png", dpi=180)
    plt.close()

    if not cross_df.empty:
        adapter_cols = [c for c in cross_df.columns if c not in {"eval_lang", "language"}]
        matrix = cross_df[adapter_cols].to_numpy(dtype=float)
        plt.figure(figsize=(10, 8))
        plt.imshow(matrix, aspect="auto")
        plt.xticks(range(len(adapter_cols)), [LANGUAGE_NAMES.get(x, x) for x in adapter_cols], rotation=45, ha="right")
        plt.yticks(range(len(cross_df)), cross_df["language"])
        plt.colorbar(label="PPL")
        plt.title("Cross-Language PPL Matrix on Separate OPUS Eval")
        plt.tight_layout()
        plt.savefig(output_dir / "cross_eval_heatmap.png", dpi=180)
        plt.close()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "separate_eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = load_run_summary(run_dir)
    model_id = args.model_id or summary["model_id"]

    model, tokenizer, device, adapter_names = load_model_and_adapters(
        run_dir=run_dir,
        model_id=model_id,
        trust_remote_code=args.trust_remote_code,
        use_bf16=args.use_bf16,
    )

    eval_langs = [lang for lang in summary["languages"] if lang in LANG_TO_OPUS]
    loaders: Dict[str, DataLoader] = {}
    dataset_meta: Dict[str, Dict[str, Any]] = {}
    for lang in eval_langs:
        texts = build_eval_texts(
            tokenizer=tokenizer,
            lang_code=lang,
            eval_samples=args.eval_samples,
            skip_samples=args.skip_samples,
            max_length=args.max_length,
        )
        ds = tokenize_eval_texts(tokenizer, texts, args.max_length)
        loaders[lang] = make_loader(ds, tokenizer, args.batch_size, args.num_workers)
        dataset_meta[lang] = {
            "language": LANGUAGE_NAMES.get(lang, lang),
            "source": f"Helsinki-NLP/opus-100::{LANG_TO_OPUS[lang][0]}",
            "field": LANG_TO_OPUS[lang][1],
            "n_eval": len(ds),
        }

    summary_rows: List[Dict[str, Any]] = []
    cross_rows: List[Dict[str, Any]] = []
    for lang in eval_langs:
        loader = loaders[lang]
        base_ppl = evaluate_loader(model, loader, device, adapter_name=None)
        own_ppl = evaluate_loader(model, loader, device, adapter_name=lang)
        row = {
            "lang": lang,
            "language": LANGUAGE_NAMES.get(lang, lang),
            "base_ppl": base_ppl,
            "adapted_ppl": own_ppl,
            "absolute_gain": base_ppl - own_ppl,
            "improvement_pct": ((base_ppl - own_ppl) / base_ppl * 100.0) if base_ppl > 0 else 0.0,
            "n_eval": dataset_meta[lang]["n_eval"],
            "source": dataset_meta[lang]["source"],
        }
        summary_rows.append(row)

        if not args.skip_cross_eval:
            cross_row: Dict[str, Any] = {
                "eval_lang": lang,
                "language": LANGUAGE_NAMES.get(lang, lang),
            }
            for adapter_name in adapter_names:
                cross_row[adapter_name] = evaluate_loader(model, loader, device, adapter_name=adapter_name)
            cross_rows.append(cross_row)

    summary_df = pd.DataFrame(summary_rows).sort_values("adapted_ppl")
    cross_df = pd.DataFrame(cross_rows)

    summary_df.to_csv(output_dir / "separate_eval_summary.csv", index=False)
    if not cross_df.empty:
        cross_df.to_csv(output_dir / "separate_cross_eval_matrix.csv", index=False)

    write_json(output_dir / "separate_eval_dataset_meta.json", dataset_meta)
    write_json(output_dir / "separate_eval_summary.json", summary_rows)
    if cross_rows:
        write_json(output_dir / "separate_cross_eval_matrix.json", cross_rows)

    plot_eval_summary(output_dir, summary_df, cross_df)

    print(json.dumps(
        {
            "run_dir": str(run_dir),
            "output_dir": str(output_dir),
            "eval_samples": args.eval_samples,
            "languages": eval_langs,
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
