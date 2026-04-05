#!/usr/bin/env python
"""Evaluate Step 5 adapters on an aligned FLORES-200 inventory.

This evaluator uses a common set of row indices across all languages, then
compares the frozen base model against the final routed adapter for each
language using normalized negative log-likelihood metrics.

Important caveat:
- For the FLORES-2K training runs, this is in-domain evaluation because the
  training setup already consumed almost all available FLORES rows.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import torch
from datasets import Dataset, concatenate_datasets, load_dataset
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

FLORES_MIRROR_LANG_MAP = {
    "ben_Beng": "ben_Beng",
    "cmn_Hans": "zho_Hans",
    "eng_Latn": "eng_Latn",
    "fra_Latn": "fra_Latn",
    "hin_Deva": "hin_Deva",
    "mar_Deva": "mar_Deva",
    "nld_Latn": "nld_Latn",
    "pol_Latn": "pol_Latn",
    "urd_Arab": "urd_Arab",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Step 5 adapters on aligned FLORES with normalized NLL metrics")
    p.add_argument("--run-dir", type=str, required=True, help="Step 5 output directory containing final/ and run_summary.json")
    p.add_argument("--output-dir", type=str, default="", help="Directory to write eval artifacts. Defaults to <run-dir>/aligned_eval_flores_norm")
    p.add_argument("--model-id", type=str, default="", help="Override base model id")
    p.add_argument("--eval-samples", type=int, default=300)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use-bf16", action="store_true")
    p.add_argument("--trust-remote-code", action="store_true")
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
    for child in adapter_dir.iterdir():
        if child.is_dir() and (child / "adapter_config.json").exists():
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
    first_name = adapter_dirs[0].name.replace("adapter_", "", 1)
    model = PeftModel.from_pretrained(base, str(first), adapter_name=first_name)
    adapter_names = [first_name]
    for adapter_dir in adapter_dirs[1:]:
        resolved = resolve_adapter_dir(adapter_dir)
        name = adapter_dir.name.replace("adapter_", "", 1)
        model.load_adapter(str(resolved), adapter_name=name)
        adapter_names.append(name)
    return model, tokenizer, device, adapter_names


def build_flores_concat() -> Dataset:
    ds = load_dataset("yash9439/flores200")
    return concatenate_datasets([ds["dev"], ds["devtest"]])


def build_common_index_set(
    ds: Dataset,
    tokenizer,
    lang_codes: List[str],
    eval_samples: int,
    max_length: int,
    seed: int,
) -> Tuple[List[int], Dict[str, Any]]:
    valid_indices: List[int] = []
    per_lang_valid = {lc: 0 for lc in lang_codes}

    for idx in range(len(ds)):
        ok = True
        for lc in lang_codes:
            col = FLORES_MIRROR_LANG_MAP[lc]
            text = ds[idx][col]
            if not isinstance(text, str) or not text.strip():
                ok = False
                break
            ids = tokenizer(text.strip(), truncation=False, padding=False)["input_ids"]
            if len(ids) <= 8 or len(ids) > max_length:
                ok = False
                break
        if ok:
            valid_indices.append(idx)
            for lc in lang_codes:
                per_lang_valid[lc] += 1

    if len(valid_indices) < eval_samples:
        raise RuntimeError(f"Only found {len(valid_indices)} common valid FLORES rows; need {eval_samples}")

    rng = random.Random(seed)
    chosen = sorted(rng.sample(valid_indices, eval_samples))
    meta = {
        "dataset": "yash9439/flores200::dev+devtest",
        "n_total_rows": len(ds),
        "n_common_valid_rows": len(valid_indices),
        "selected_indices": chosen,
        "per_language_valid_rows": per_lang_valid,
    }
    return chosen, meta


def build_eval_rows(ds: Dataset, lang_code: str, indices: List[int]) -> List[Dict[str, Any]]:
    col = FLORES_MIRROR_LANG_MAP[lang_code]
    rows: List[Dict[str, Any]] = []
    for idx in indices:
        text = str(ds[idx][col]).strip()
        rows.append(
            {
                "row_idx": idx,
                "text": text,
                "n_chars": len(text),
                "n_bytes": len(text.encode("utf-8")),
            }
        )
    return rows


def tokenize_eval_rows(tokenizer, rows: List[Dict[str, Any]], max_length: int) -> Dataset:
    raw = Dataset.from_list(rows)

    def _tok(batch):
        tok = tokenizer(batch["text"], truncation=True, max_length=max_length, padding=False)
        tok["labels"] = [ids[:] for ids in tok["input_ids"]]
        tok["n_chars"] = batch["n_chars"]
        tok["n_bytes"] = batch["n_bytes"]
        tok["row_idx"] = batch["row_idx"]
        return tok

    return raw.map(_tok, batched=True, remove_columns=["text"])


def make_collator(tokenizer):
    pad_id = tokenizer.pad_token_id

    def _collate(features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids = []
        attention_mask = []
        labels = []
        n_chars = []
        n_bytes = []
        row_idx = []
        for f in features:
            ids = list(f["input_ids"])
            mask = list(f["attention_mask"])
            labs = list(f["labels"])
            pad_len = max_len - len(ids)
            input_ids.append(ids + [pad_id] * pad_len)
            attention_mask.append(mask + [0] * pad_len)
            labels.append(labs + [-100] * pad_len)
            n_chars.append(int(f["n_chars"]))
            n_bytes.append(int(f["n_bytes"]))
            row_idx.append(int(f["row_idx"]))
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "n_chars": torch.tensor(n_chars, dtype=torch.long),
            "n_bytes": torch.tensor(n_bytes, dtype=torch.long),
            "row_idx": torch.tensor(row_idx, dtype=torch.long),
        }

    return _collate


def make_loader(ds: Dataset, tokenizer, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=make_collator(tokenizer),
        drop_last=False,
        num_workers=num_workers,
    )


def evaluate_loader(model, loader: DataLoader, device: torch.device, adapter_name: str | None) -> Dict[str, float]:
    model.eval()
    total_nll = 0.0
    total_pred_tokens = 0
    total_chars = 0
    total_bytes = 0

    if adapter_name is None:
        adapter_ctx = model.disable_adapter()
    else:
        model.set_adapter(adapter_name)
        adapter_ctx = nullcontext()

    with adapter_ctx:
        with torch.no_grad():
            for batch in loader:
                n_chars = int(batch["n_chars"].sum().item())
                n_bytes = int(batch["n_bytes"].sum().item())
                labels = batch["labels"]
                valid = (labels != -100).sum(dim=1)
                pred_tokens = int((valid - 1).clamp(min=0).sum().item())

                forward_batch = {k: v.to(device) for k, v in batch.items() if k in {"input_ids", "attention_mask", "labels"}}
                out = model(**forward_batch)

                total_nll += float(out.loss.detach().float().item()) * pred_tokens
                total_pred_tokens += pred_tokens
                total_chars += n_chars
                total_bytes += n_bytes

    nll_per_token = total_nll / max(total_pred_tokens, 1)
    nll_per_char = total_nll / max(total_chars, 1)
    nll_per_byte = total_nll / max(total_bytes, 1)
    return {
        "nll_per_token": nll_per_token,
        "ppl": float(math.exp(nll_per_token)),
        "nll_per_char": nll_per_char,
        "nll_per_byte": nll_per_byte,
        "bits_per_char": nll_per_char / math.log(2.0),
        "bits_per_byte": nll_per_byte / math.log(2.0),
        "pred_tokens": total_pred_tokens,
        "chars": total_chars,
        "bytes": total_bytes,
    }


def plot_summary(output_dir: Path, summary_df: pd.DataFrame) -> None:
    ordered = summary_df.sort_values("adapted_bits_per_byte")

    plt.figure(figsize=(11, 6))
    y = range(len(ordered))
    plt.barh([i + 0.18 for i in y], ordered["base_bits_per_byte"], height=0.35, label="Base")
    plt.barh([i - 0.18 for i in y], ordered["adapted_bits_per_byte"], height=0.35, label="Final Adapter")
    plt.yticks(list(y), ordered["language"])
    plt.xlabel("Bits per Byte")
    plt.title("Aligned FLORES Eval: Base vs Final Adapter")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "base_vs_adapter_bits_per_byte.png", dpi=180)
    plt.close()

    plt.figure(figsize=(11, 6))
    plt.barh(ordered["language"], ordered["byte_improvement_pct"])
    plt.xlabel("Improvement over Base (%)")
    plt.title("Aligned FLORES Eval: Relative Improvement (Bits per Byte)")
    plt.tight_layout()
    plt.savefig(output_dir / "bits_per_byte_improvement.png", dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "aligned_eval_flores_norm"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = load_run_summary(run_dir)
    model_id = args.model_id or summary["model_id"]
    lang_codes = [lang for lang in summary["languages"] if lang in FLORES_MIRROR_LANG_MAP]

    model, tokenizer, device, adapter_names = load_model_and_adapters(
        run_dir=run_dir,
        model_id=model_id,
        trust_remote_code=args.trust_remote_code,
        use_bf16=args.use_bf16,
    )
    adapter_set = set(adapter_names)
    missing = [lang for lang in lang_codes if lang not in adapter_set]
    if missing:
        raise RuntimeError(f"Missing adapters for languages: {missing}")

    ds = build_flores_concat()
    indices, eval_meta = build_common_index_set(
        ds=ds,
        tokenizer=tokenizer,
        lang_codes=lang_codes,
        eval_samples=args.eval_samples,
        max_length=args.max_length,
        seed=args.seed,
    )

    loaders: Dict[str, DataLoader] = {}
    for lang in lang_codes:
        rows = build_eval_rows(ds, lang, indices)
        tok_ds = tokenize_eval_rows(tokenizer, rows, args.max_length)
        loaders[lang] = make_loader(tok_ds, tokenizer, args.batch_size, args.num_workers)

    summary_rows: List[Dict[str, Any]] = []
    for lang in lang_codes:
        base_metrics = evaluate_loader(model, loaders[lang], device, adapter_name=None)
        adapted_metrics = evaluate_loader(model, loaders[lang], device, adapter_name=lang)

        row = {
            "lang": lang,
            "language": LANGUAGE_NAMES.get(lang, lang),
            "n_eval": args.eval_samples,
            "base_ppl": base_metrics["ppl"],
            "adapted_ppl": adapted_metrics["ppl"],
            "base_nll_per_char": base_metrics["nll_per_char"],
            "adapted_nll_per_char": adapted_metrics["nll_per_char"],
            "base_nll_per_byte": base_metrics["nll_per_byte"],
            "adapted_nll_per_byte": adapted_metrics["nll_per_byte"],
            "base_bits_per_char": base_metrics["bits_per_char"],
            "adapted_bits_per_char": adapted_metrics["bits_per_char"],
            "base_bits_per_byte": base_metrics["bits_per_byte"],
            "adapted_bits_per_byte": adapted_metrics["bits_per_byte"],
            "char_improvement_pct": (
                (base_metrics["nll_per_char"] - adapted_metrics["nll_per_char"]) / base_metrics["nll_per_char"] * 100.0
                if base_metrics["nll_per_char"] > 0
                else 0.0
            ),
            "byte_improvement_pct": (
                (base_metrics["nll_per_byte"] - adapted_metrics["nll_per_byte"]) / base_metrics["nll_per_byte"] * 100.0
                if base_metrics["nll_per_byte"] > 0
                else 0.0
            ),
            "ppl_improvement_pct": (
                (base_metrics["ppl"] - adapted_metrics["ppl"]) / base_metrics["ppl"] * 100.0
                if base_metrics["ppl"] > 0
                else 0.0
            ),
        }
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).sort_values("adapted_bits_per_byte")
    summary_df.to_csv(output_dir / "aligned_flores_base_vs_adapter.csv", index=False)
    write_json(output_dir / "aligned_flores_eval_meta.json", eval_meta)
    write_json(output_dir / "aligned_flores_base_vs_adapter.json", summary_rows)
    plot_summary(output_dir, summary_df)

    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "output_dir": str(output_dir),
                "eval_samples": args.eval_samples,
                "languages": lang_codes,
                "n_common_valid_rows": eval_meta["n_common_valid_rows"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
