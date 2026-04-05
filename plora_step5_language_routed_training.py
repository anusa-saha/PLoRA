#!/usr/bin/env python
"""PLoRA Step 5: Language-routed training objective.

Implements paper section 1.6:
- Step 5.1: Per-language likelihood training with routed adapters.
- Step 5.2: Optional geometry-aware regularization (stable-rank proxy).

Designed for Cloudexe/remote GPU execution. Consumes Step 4 output JSON.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import logging
import math
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

LOGGER = logging.getLogger("plora.step5")


# FLORES/NLLB -> OPUS-100 fallback mapping (only used when needed)
FLORES_TO_OPUS = {
    "eng_Latn": "en",
    "fra_Latn": "fr",
    "cmn_Hans": "zh",
    "urd_Arab": "ur",
    "hin_Deva": "hi",
    "ben_Beng": "bn",
    "mar_Deva": "mr",
    "nld_Latn": "nl",
    "pol_Latn": "pl",
    # likely unavailable in OPUS-100:
    # awa_Deva, snd_Arab, azb_Arab
}

ATTN_MODULE_HINTS = {"q_proj", "k_proj", "v_proj", "o_proj"}
MLP_MODULE_HINTS = {"up_proj", "down_proj", "gate_proj"}
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
NATURAL_WEB_CONFIG_MAP = {
    "eng_Latn": ("HuggingFaceFW/fineweb", None),
    "fra_Latn": ("HuggingFaceFW/fineweb-2", "fra_Latn"),
    "cmn_Hans": ("HuggingFaceFW/fineweb-2", "cmn_Hani"),
    "urd_Arab": ("HuggingFaceFW/fineweb-2", "urd_Arab"),
    "hin_Deva": ("HuggingFaceFW/fineweb-2", "hin_Deva"),
    "ben_Beng": ("HuggingFaceFW/fineweb-2", "ben_Beng"),
    "mar_Deva": ("HuggingFaceFW/fineweb-2", "mar_Deva"),
    "nld_Latn": ("HuggingFaceFW/fineweb-2", "nld_Latn"),
    "pol_Latn": ("HuggingFaceFW/fineweb-2", "pol_Latn"),
}


@dataclass
class LangData:
    train_loader: DataLoader
    eval_loader: DataLoader
    num_train: int
    num_eval: int
    dataset_source: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PLoRA Step 5 language-routed training")
    p.add_argument("--rank-json", type=str, default="plora_step4_rank_budgets.json")
    p.add_argument("--model-id", type=str, default="Qwen/Qwen3-4B-Base", help="Base model id for Step 5 training")
    p.add_argument("--budget-key", type=str, default="fair_budget", choices=["fair_budget", "equal_budget"])
    p.add_argument("--output-dir", type=str, default="outputs/plora_step5")

    p.add_argument("--max-train-samples", type=int, default=10000)
    p.add_argument("--max-eval-samples", type=int, default=1000)
    p.add_argument("--train-batch-size", type=int, default=2)
    p.add_argument("--eval-batch-size", type=int, default=2)
    p.add_argument("--grad-accum-steps", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--save-every", type=int, default=500)

    p.add_argument("--lambda-chan", type=float, default=0.0, help="Optional Step 5.2 regularizer weight")
    p.add_argument("--power-iters", type=int, default=3, help="Power iterations for sigma_max estimate")
    p.add_argument(
        "--dataset-kind",
        type=str,
        default="natural_web_mix",
        choices=["opus_or_flores", "flores200_mirror", "natural_web_mix"],
        help="Corpus source for Step 5 training.",
    )

    p.add_argument("--target-languages", type=str, default="", help="Comma-separated language codes")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--use-bf16", action="store_true")
    p.add_argument("--skip-cross-eval", action="store_true", help="Disable final adapter-vs-language PPL matrix.")
    p.add_argument("--skip-report-plots", action="store_true", help="Disable report PNG generation.")
    p.add_argument("--dry-run", action="store_true", help="Load data/model and run 2 optimizer steps")
    return p.parse_args()


def load_rank_payload(path: str) -> dict:
    with open(path, "r", encoding="utf-8-sig") as f:
        payload = json.load(f)
    required = ["metadata", "support_sets", "fair_budget", "equal_budget"]
    for key in required:
        if key not in payload:
            raise ValueError(f"Missing key in rank JSON: {key}")
    return payload


def pick_opus_config(lang_code: str, available: set[str]) -> str | None:
    if lang_code not in FLORES_TO_OPUS:
        return None
    iso = FLORES_TO_OPUS[lang_code]
    if iso == "en":
        for fallback in ("en-fr", "fr-en", "en-hi", "hi-en"):
            if fallback in available:
                return fallback
        return None
    a = f"{iso}-en"
    b = f"en-{iso}"
    if a in available:
        return a
    if b in available:
        return b
    return None


def build_monolingual_texts(lang_code: str, max_samples: int) -> Tuple[List[str], str]:
    """Use multilingual translation corpora with broad online availability.

    Priority:
      1) OPUS-100 (open, large multilingual parallel corpus).
      2) FLORES+ per-language text as fallback for uncovered language codes.
    """

    try:
        from datasets import get_dataset_config_names

        opus_configs = set(get_dataset_config_names("Helsinki-NLP/opus-100"))
    except Exception as e:  # pragma: no cover
        LOGGER.warning("Could not list OPUS-100 configs: %s", e)
        opus_configs = set()

    opus_pair = pick_opus_config(lang_code, opus_configs)
    if opus_pair:
        ds = load_dataset("Helsinki-NLP/opus-100", opus_pair, split="train", streaming=True)
        iso = FLORES_TO_OPUS[lang_code]
        texts = []
        for row in ds:
            translation = row.get("translation", {})
            t = translation.get(iso, "")
            if iso == "en" and not t:
                for key in ("en", "source", "target"):
                    val = translation.get(key, "")
                    if isinstance(val, str) and val.strip():
                        t = val
                        break
            if isinstance(t, str) and t.strip():
                texts.append(t.strip())
            if len(texts) >= max_samples:
                break
        if texts:
            return texts, f"Helsinki-NLP/opus-100::{opus_pair}"

    # Fallback used in previous project steps; may require authenticated HF access.
    try:
        ds = load_dataset("openlanguagedata/flores_plus", lang_code, split="devtest")
        text_col = "text" if "text" in ds.column_names else "sentence"
        texts = []
        for row in ds:
            t = row.get(text_col, "")
            if isinstance(t, str) and t.strip():
                texts.append(t.strip())
            if len(texts) >= max_samples:
                break
        if texts:
            return texts, f"openlanguagedata/flores_plus::{lang_code}"
    except Exception as e:
        LOGGER.warning("FLORES+ fallback unavailable for %s: %s", lang_code, e)

    raise RuntimeError(
        f"No multilingual translation dataset available for {lang_code}. "
        "Checked OPUS-100 and FLORES+."
    )


def build_natural_web_texts(lang_code: str, max_samples: int) -> Tuple[List[str], str]:
    """Load natural monolingual web text for Step 5 causal LM training.

    This avoids benchmark-style translation corpora for training and instead uses
    cleaned web corpora:
      - English: FineWeb
      - Other supported languages: FineWeb2 language-specific subsets
    """
    spec = NATURAL_WEB_CONFIG_MAP.get(lang_code)
    if spec is None:
        raise RuntimeError(f"No natural web corpus mapping configured for {lang_code}")

    dataset_name, config_name = spec
    if config_name is None:
        ds = load_dataset(dataset_name, split="train", streaming=True)
    else:
        ds = load_dataset(dataset_name, config_name, split="train", streaming=True)

    texts: List[str] = []
    seen = set()
    for row in ds:
        text = row.get("text", "")
        if not isinstance(text, str):
            continue
        text = text.strip()
        if not text or len(text) < 32:
            continue
        # Avoid repeated boilerplate and exact duplicates in the first chunk.
        if text in seen:
            continue
        seen.add(text)
        texts.append(text)
        if len(texts) >= max_samples:
            break

    if not texts:
        raise RuntimeError(f"No usable natural web texts loaded for {lang_code} from {dataset_name}::{config_name}")

    source = dataset_name if config_name is None else f"{dataset_name}::{config_name}"
    return texts, source


def build_flores200_exact_texts(lang_code: str, train_samples: int, eval_samples: int) -> Tuple[List[str], str]:
    """Load a fixed-count aligned corpus from a public FLORES-200 parquet mirror.

    This keeps language sample counts identical across languages. Note that FLORES
    is conventionally used for evaluation; using it for training changes that role.
    """
    column = FLORES_MIRROR_LANG_MAP.get(lang_code, lang_code)
    ds = load_dataset("yash9439/flores200")
    total_needed = train_samples + eval_samples
    texts: List[str] = []
    for split in ("dev", "devtest"):
        split_ds = ds[split]
        if column not in split_ds.column_names:
            raise RuntimeError(f"FLORES-200 mirror missing column {column} for {lang_code}")
        for text in split_ds[column]:
            if isinstance(text, str) and text.strip():
                texts.append(text.strip())
            if len(texts) >= total_needed:
                break
        if len(texts) >= total_needed:
            break
    if len(texts) < total_needed:
        raise RuntimeError(
            f"FLORES-200 mirror has only {len(texts)} usable texts for {lang_code}; "
            f"need {total_needed}"
        )
    return texts[:total_needed], f"yash9439/flores200::{column}"


def tokenize_texts(texts: List[str], tokenizer, max_len: int) -> Dataset:
    raw = Dataset.from_dict({"text": texts})

    def _tok(batch):
        tok = tokenizer(batch["text"], truncation=True, max_length=max_len, padding=False)
        tok["labels"] = [ids[:] for ids in tok["input_ids"]]
        return tok

    tokenized = raw.map(_tok, batched=True, remove_columns=["text"])
    tokenized = tokenized.filter(lambda x: len(x["input_ids"]) > 8)
    return tokenized


def make_causal_lm_collator(tokenizer):
    pad_id = tokenizer.pad_token_id

    def _collate(features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_feature_len = max(len(f["input_ids"]) for f in features)
        input_ids = []
        attention_mask = []
        labels = []
        for f in features:
            ids = list(f["input_ids"])
            mask = list(f["attention_mask"])
            labs = list(f["labels"])
            pad_len = max_feature_len - len(ids)
            input_ids.append(ids + [pad_id] * pad_len)
            attention_mask.append(mask + [0] * pad_len)
            labels.append(labs + [-100] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    return _collate


def split_dataset(ds: Dataset, max_eval: int, seed: int) -> Tuple[Dataset, Dataset]:
    if len(ds) <= max_eval + 1:
        cut = max(1, len(ds) // 10)
        eval_ds = ds.select(range(cut))
        train_ds = ds.select(range(cut, len(ds)))
        return train_ds, eval_ds

    ds = ds.shuffle(seed=seed)
    eval_ds = ds.select(range(max_eval))
    train_ds = ds.select(range(max_eval, len(ds)))
    return train_ds, eval_ds


def create_dataloaders(
    tokenizer,
    lang_codes: Iterable[str],
    max_train: int,
    max_eval: int,
    max_len: int,
    train_bs: int,
    eval_bs: int,
    seed: int,
    dataset_kind: str,
) -> Dict[str, LangData]:
    result: Dict[str, LangData] = {}
    collator = make_causal_lm_collator(tokenizer)
    for lc in lang_codes:
        LOGGER.info("Loading multilingual corpus for %s", lc)
        if dataset_kind == "flores200_mirror":
            texts, source = build_flores200_exact_texts(lc, max_train, max_eval)
        elif dataset_kind == "natural_web_mix":
            texts, source = build_natural_web_texts(lc, max_train + max_eval)
        else:
            texts, source = build_monolingual_texts(lc, max_train + max_eval)
        tok_ds = tokenize_texts(texts, tokenizer, max_len)
        if dataset_kind == "flores200_mirror":
            if len(tok_ds) < max_train:
                raise RuntimeError(
                    f"After tokenization/filtering, {lc} has only {len(tok_ds)} items; "
                    f"need at least {max_train} for exact-count FLORES mode."
                )
            train_ds = tok_ds.select(range(max_train))
            eval_count = min(max_eval, len(tok_ds) - max_train)
            eval_ds = tok_ds.select(range(max_train, max_train + eval_count))
            LOGGER.info(
                "%s exact-count FLORES mode -> requested train=%d eval=%d | actual eval=%d after tokenization",
                lc,
                max_train,
                max_eval,
                eval_count,
            )
        else:
            train_ds, eval_ds = split_dataset(tok_ds, max_eval=max_eval, seed=seed)

        train_loader = DataLoader(
            train_ds,
            batch_size=train_bs,
            shuffle=True,
            collate_fn=collator,
            drop_last=False,
        )
        eval_loader = DataLoader(
            eval_ds,
            batch_size=eval_bs,
            shuffle=False,
            collate_fn=collator,
            drop_last=False,
        )
        result[lc] = LangData(
            train_loader=train_loader,
            eval_loader=eval_loader,
            num_train=len(train_ds),
            num_eval=len(eval_ds),
            dataset_source=source,
        )
        LOGGER.info(
            "%s -> %s | train=%d eval=%d",
            lc,
            source,
            len(train_ds),
            len(eval_ds),
        )
    return result


def _find_modules_for_layer(
    module_names: List[str], layer_idx: int, module_hints: Iterable[str]
) -> Dict[str, str]:
    out = {}
    for hint in module_hints:
        rx = re.compile(rf"(^|\.)layers\.{layer_idx}(\.|$).+\.{re.escape(hint)}$")
        for name in module_names:
            if rx.search(name):
                out[hint] = name
                break
    return out


def build_adapter_module_map(module_names: List[str], support: List[int], target_hints: List[str]) -> Dict[str, int]:
    rank_pattern: Dict[str, int] = {}
    for ell in support:
        found = _find_modules_for_layer(module_names, ell, target_hints)
        for hint, full in found.items():
            if full not in rank_pattern:
                rank_pattern[full] = 0  # placeholder; overwritten per language ranks
    return rank_pattern


def freeze_backbone(model) -> None:
    for n, p in model.named_parameters():
        if "lora_" not in n:
            p.requires_grad = False


def set_trainable_adapter(model, adapter_name: str) -> List[torch.nn.Parameter]:
    trainable = []
    for n, p in model.named_parameters():
        if "lora_" not in n:
            p.requires_grad = False
            continue
        on = f".{adapter_name}." in n
        p.requires_grad = on
        if on:
            trainable.append(p)
    return trainable


def power_sigma_max(x: torch.Tensor, iters: int = 3) -> torch.Tensor:
    """Estimate spectral norm via power iteration on centered matrix x [N, D]."""
    if x.ndim != 2:
        raise ValueError("x must be 2D")
    v = torch.randn(x.shape[1], device=x.device, dtype=x.dtype)
    v = v / (v.norm() + 1e-8)
    for _ in range(iters):
        u = x @ v
        u = u / (u.norm() + 1e-8)
        v = x.t() @ u
        v = v / (v.norm() + 1e-8)
    sigma = torch.norm(x @ v)
    return sigma


def stable_rank_penalty(
    hidden_states: Tuple[torch.Tensor, ...],
    support_layers: List[int],
    target_ranks: Dict[int, int],
    power_iters: int,
) -> torch.Tensor:
    penalties = []
    # hidden_states[0] is embeddings; layer l output is hidden_states[l+1]
    for ell in support_layers:
        hs = hidden_states[ell + 1]
        h = hs.reshape(-1, hs.shape[-1])
        h = h - h.mean(dim=0, keepdim=True)
        fro2 = torch.sum(h * h)
        sigma = power_sigma_max(h, iters=power_iters)
        sr = fro2 / (sigma * sigma + 1e-8)
        target = float(max(1, target_ranks.get(ell, 1)))
        penalties.append((torch.log(sr + 1e-8) - math.log(target + 1e-8)) ** 2)
    if not penalties:
        return torch.tensor(0.0, device=hidden_states[0].device)
    return torch.stack(penalties).mean()


def evaluate_ppl(model, loader: DataLoader, device: torch.device, adapter_name: str) -> float:
    model.eval()
    model.set_adapter(adapter_name)
    losses = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            losses.append(out.loss.detach().float().item())
    if not losses:
        return float("inf")
    return float(math.exp(sum(losses) / len(losses)))


def evaluate_ppl_matrix(
    model,
    datasets_by_lang: Dict[str, LangData],
    device: torch.device,
    adapter_langs: List[str],
    eval_langs: List[str],
) -> Dict[str, Dict[str, float]]:
    matrix: Dict[str, Dict[str, float]] = {}
    for eval_lang in eval_langs:
        row: Dict[str, float] = {}
        loader = datasets_by_lang[eval_lang].eval_loader
        for adapter_lang in adapter_langs:
            row[adapter_lang] = evaluate_ppl(model, loader, device, adapter_lang)
        matrix[eval_lang] = row
    return matrix


def write_json(path: Path, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize_train_log(train_log: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_lang: Dict[str, List[float]] = {}
    for row in train_log:
        by_lang.setdefault(row["lang"], []).append(float(row["loss"]))
    summary_rows: List[Dict[str, Any]] = []
    for lang, losses in sorted(by_lang.items()):
        first = losses[:20]
        last = losses[-20:]
        first_mean = sum(first) / len(first)
        last_mean = sum(last) / len(last)
        improve = 0.0
        if first_mean > 0:
            improve = 100.0 * (first_mean - last_mean) / first_mean
        summary_rows.append(
            {
                "lang": lang,
                "steps_seen": len(losses),
                "mean_loss": sum(losses) / len(losses),
                "first20_mean_loss": first_mean,
                "last20_mean_loss": last_mean,
                "final_logged_loss": losses[-1],
                "improvement_pct": improve,
            }
        )
    return summary_rows


def build_budget_summary(
    language_names: Dict[str, str],
    support_sets: Dict[str, List[int]],
    rank_map: Dict[str, Dict[str, int]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for lang in sorted(rank_map):
        ranks = [int(v) for _, v in sorted(rank_map[lang].items(), key=lambda kv: int(kv[0]))]
        rows.append(
            {
                "lang": lang,
                "language": language_names.get(lang, lang),
                "support_layers": len(support_sets.get(lang, [])),
                "rank_sum": sum(ranks),
                "max_rank": max(ranks) if ranks else 0,
                "min_rank": min(ranks) if ranks else 0,
            }
        )
    return rows


def cross_eval_to_rows(
    language_names: Dict[str, str],
    matrix: Dict[str, Dict[str, float]],
    adapter_langs: List[str],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for eval_lang, row in matrix.items():
        out: Dict[str, Any] = {
            "eval_lang": eval_lang,
            "eval_language": language_names.get(eval_lang, eval_lang),
        }
        for adapter_lang in adapter_langs:
            out[adapter_lang] = row.get(adapter_lang, float("inf"))
        rows.append(out)
    return rows


def summarize_cross_eval(
    language_names: Dict[str, str],
    matrix: Dict[str, Dict[str, float]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for eval_lang, row in sorted(matrix.items()):
        own = float(row.get(eval_lang, float("inf")))
        best_adapter, best_ppl = min(row.items(), key=lambda kv: kv[1])
        rows.append(
            {
                "lang": eval_lang,
                "language": language_names.get(eval_lang, eval_lang),
                "own_adapter_ppl": own,
                "best_adapter": best_adapter,
                "best_adapter_ppl": float(best_ppl),
                "own_minus_best": own - float(best_ppl),
            }
        )
    return rows


def render_report_plots(
    out_dir: Path,
    train_log: List[Dict[str, Any]],
    eval_history: List[Dict[str, Any]],
    cross_eval_rows: List[Dict[str, Any]],
    diagonal_rows: List[Dict[str, Any]],
    language_names: Dict[str, str],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        LOGGER.warning("Skipping plots because matplotlib is unavailable: %s", e)
        return

    by_lang: Dict[str, List[Tuple[int, float]]] = {}
    for row in train_log:
        by_lang.setdefault(row["lang"], []).append((int(row["step"]), float(row["loss"])))

    plt.figure(figsize=(12, 7))
    for lang, points in sorted(by_lang.items()):
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        smoothed = []
        for idx in range(len(ys)):
            lo = max(0, idx - 24)
            window = ys[lo : idx + 1]
            smoothed.append(sum(window) / len(window))
        plt.plot(xs, smoothed, label=language_names.get(lang, lang), linewidth=1.6)
    plt.xlabel("Global Step")
    plt.ylabel("Smoothed Train Loss")
    plt.title("Step 5 Train Loss by Language")
    plt.grid(alpha=0.25)
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "train_loss_by_language.png", dpi=180)
    plt.close()

    if eval_history:
        plt.figure(figsize=(12, 7))
        metric_keys = [k for k in eval_history[0] if k.startswith("ppl_")]
        for key in sorted(metric_keys):
            lang = key.replace("ppl_", "", 1)
            xs = [int(row["step"]) for row in eval_history]
            ys = [float(row[key]) for row in eval_history]
            plt.plot(xs, ys, marker="o", label=language_names.get(lang, lang), linewidth=1.6)
        plt.xlabel("Global Step")
        plt.ylabel("Perplexity")
        plt.title("Step 5 Diagonal Eval PPL")
        plt.grid(alpha=0.25)
        plt.legend(ncol=3, fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / "eval_ppl_over_time.png", dpi=180)
        plt.close()

    if diagonal_rows:
        plt.figure(figsize=(10, 6))
        labels = [row["language"] for row in diagonal_rows]
        vals = [float(row["own_adapter_ppl"]) for row in diagonal_rows]
        plt.barh(labels, vals)
        plt.xlabel("Final Own-Adapter PPL")
        plt.title("Final Diagonal Eval")
        plt.tight_layout()
        plt.savefig(out_dir / "final_diagonal_eval.png", dpi=180)
        plt.close()

    if cross_eval_rows:
        adapter_langs = [k for k in cross_eval_rows[0].keys() if k not in {"eval_lang", "eval_language"}]
        matrix = [[float(row[adapter]) for adapter in adapter_langs] for row in cross_eval_rows]
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(matrix, aspect="auto")
        ax.set_xticks(range(len(adapter_langs)))
        ax.set_xticklabels([language_names.get(x, x) for x in adapter_langs], rotation=45, ha="right")
        ax.set_yticks(range(len(cross_eval_rows)))
        ax.set_yticklabels([row["eval_language"] for row in cross_eval_rows])
        ax.set_title("Final Cross-Language PPL Matrix")
        fig.colorbar(im, ax=ax, shrink=0.85)
        plt.tight_layout()
        plt.savefig(out_dir / "cross_eval_heatmap.png", dpi=180)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    set_seed(args.seed)
    random.seed(args.seed)

    payload = load_rank_payload(args.rank_json)
    model_id = args.model_id or payload["metadata"].get("model_id", "Qwen/Qwen3-4B-Base")
    support_sets: Dict[str, List[int]] = payload["support_sets"]
    rank_map: Dict[str, Dict[str, int]] = payload[args.budget_key]
    language_names: Dict[str, str] = payload.get("metadata", {}).get("languages", {})

    lang_codes = sorted(set(support_sets) & set(rank_map))
    if args.target_languages.strip():
        chosen = {x.strip() for x in args.target_languages.split(",") if x.strip()}
        lang_codes = [lc for lc in lang_codes if lc in chosen]

    if not lang_codes:
        raise RuntimeError("No languages available after filtering.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Model: %s", model_id)
    LOGGER.info("Languages: %s", ", ".join(lang_codes))
    LOGGER.info("Budget key: %s", args.budget_key)

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    datasets_by_lang = create_dataloaders(
        tokenizer=tokenizer,
        lang_codes=lang_codes,
        max_train=args.max_train_samples,
        max_eval=args.max_eval_samples,
        max_len=args.max_length,
        train_bs=args.train_batch_size,
        eval_bs=args.eval_batch_size,
        seed=args.seed,
        dataset_kind=args.dataset_kind,
    )

    dtype = torch.bfloat16 if (args.use_bf16 and torch.cuda.is_available()) else torch.float16
    if not torch.cuda.is_available():
        dtype = torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if not torch.cuda.is_available():
        model.to(device)

    target_hints = payload["metadata"].get("lora_targets", {}).get("qwen", [])
    if not target_hints:
        target_hints = sorted(ATTN_MODULE_HINTS | MLP_MODULE_HINTS)

    base_module_names = [n for n, _ in model.named_modules()]

    # Build adapters per language with layer-specific rank patterns.
    peft_model: PeftModel | None = None
    for i, lc in enumerate(lang_codes):
        support = [int(x) for x in support_sets[lc]]
        rank_int = {int(k): int(v) for k, v in rank_map[lc].items() if int(v) > 0}

        rank_pattern_template = build_adapter_module_map(base_module_names, support, target_hints)
        rank_pattern = {}
        alpha_pattern = {}

        for module_name in rank_pattern_template:
            m = re.search(r"layers\.(\d+)\.", module_name)
            if not m:
                continue
            ell = int(m.group(1))
            r = int(rank_int.get(ell, 0))
            if r > 0:
                rank_pattern[module_name] = r
                alpha_pattern[module_name] = max(8, 2 * r)

        if not rank_pattern:
            raise RuntimeError(f"No target modules resolved for language {lc}")

        cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=sorted(rank_pattern.keys()),
            r=max(rank_pattern.values()),
            lora_alpha=max(alpha_pattern.values()),
            rank_pattern=rank_pattern,
            alpha_pattern=alpha_pattern,
            lora_dropout=0.05,
            bias="none",
        )

        if i == 0:
            peft_model = get_peft_model(model, cfg, adapter_name=lc)
        else:
            peft_model.add_adapter(adapter_name=lc, peft_config=cfg)

    assert peft_model is not None
    model = peft_model
    freeze_backbone(model)
    model.config.use_cache = False

    optimizers: Dict[str, AdamW] = {}
    for lc in lang_codes:
        params = set_trainable_adapter(model, lc)
        if not params:
            raise RuntimeError(f"No trainable params found for adapter {lc}")
        optimizers[lc] = AdamW(params, lr=args.learning_rate, weight_decay=args.weight_decay)

    train_iters = {lc: iter(datasets_by_lang[lc].train_loader) for lc in lang_codes}
    scaler = None  # kept off for simplicity and bf16 compatibility

    train_log = []
    eval_history: List[Dict[str, Any]] = []
    global_step = 0
    pbar = tqdm(total=(2 if args.dry_run else args.max_steps), desc="step5-train", ncols=100)

    while True:
        for lc in lang_codes:
            global_step += 1
            if args.dry_run and global_step > 2:
                break
            if (not args.dry_run) and global_step > args.max_steps:
                break

            try:
                batch = next(train_iters[lc])
            except StopIteration:
                train_iters[lc] = iter(datasets_by_lang[lc].train_loader)
                batch = next(train_iters[lc])

            model.train()
            model.set_adapter(lc)
            set_trainable_adapter(model, lc)

            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch, output_hidden_states=args.lambda_chan > 0)
            loss = outputs.loss

            if args.lambda_chan > 0:
                support = [int(x) for x in support_sets[lc]]
                target_ranks = {int(k): int(v) for k, v in rank_map[lc].items()}
                reg = stable_rank_penalty(
                    outputs.hidden_states,
                    support_layers=support,
                    target_ranks=target_ranks,
                    power_iters=args.power_iters,
                )
                loss = loss + args.lambda_chan * reg

            loss = loss / args.grad_accum_steps
            loss.backward()

            if global_step % args.grad_accum_steps == 0:
                optimizers[lc].step()
                optimizers[lc].zero_grad(set_to_none=True)

            train_log.append({"step": global_step, "lang": lc, "loss": float(loss.item() * args.grad_accum_steps)})
            if global_step % args.log_every == 0:
                recent = train_log[-args.log_every :]
                avg = sum(x["loss"] for x in recent) / len(recent)
                LOGGER.info("step=%d lang=%s avg_loss=%.4f", global_step, lc, avg)

            if (not args.dry_run) and args.eval_every > 0 and (global_step % args.eval_every == 0):
                metrics: Dict[str, Any] = {"step": global_step}
                for elc in lang_codes:
                    ppl = evaluate_ppl(model, datasets_by_lang[elc].eval_loader, device, elc)
                    metrics[f"ppl_{elc}"] = ppl
                eval_history.append(metrics)
                LOGGER.info("Eval@%d -> %s", global_step, metrics)

            if (not args.dry_run) and args.save_every > 0 and (global_step % args.save_every == 0):
                ckpt = out_dir / f"checkpoint_step_{global_step}"
                ckpt.mkdir(parents=True, exist_ok=True)
                for slc in lang_codes:
                    model.set_adapter(slc)
                    model.save_pretrained(ckpt / f"adapter_{slc}", selected_adapters=[slc])
                tokenizer.save_pretrained(ckpt / "tokenizer")
                LOGGER.info("Saved checkpoint: %s", ckpt)

            pbar.update(1)

        if args.dry_run and global_step > 2:
            break
        if (not args.dry_run) and global_step > args.max_steps:
            break

    pbar.close()

    final_dir = out_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    # Save each adapter separately for routed inference.
    for lc in lang_codes:
        model.set_adapter(lc)
        model.save_pretrained(final_dir / f"adapter_{lc}", selected_adapters=[lc])
    tokenizer.save_pretrained(final_dir / "tokenizer")

    final_diag_eval: Dict[str, float] = {}
    for lc in lang_codes:
        final_diag_eval[lc] = evaluate_ppl(model, datasets_by_lang[lc].eval_loader, device, lc)

    cross_eval_matrix: Dict[str, Dict[str, float]] = {}
    if not args.skip_cross_eval:
        cross_eval_matrix = evaluate_ppl_matrix(
            model=model,
            datasets_by_lang=datasets_by_lang,
            device=device,
            adapter_langs=lang_codes,
            eval_langs=lang_codes,
        )

    train_loss_summary = summarize_train_log(train_log)
    budget_summary = build_budget_summary(language_names, support_sets, rank_map)
    cross_eval_rows = cross_eval_to_rows(language_names, cross_eval_matrix, lang_codes) if cross_eval_matrix else []
    diagonal_eval_rows = (
        summarize_cross_eval(language_names, cross_eval_matrix)
        if cross_eval_matrix
        else [
            {
                "lang": lc,
                "language": language_names.get(lc, lc),
                "own_adapter_ppl": final_diag_eval[lc],
                "best_adapter": lc,
                "best_adapter_ppl": final_diag_eval[lc],
                "own_minus_best": 0.0,
            }
            for lc in lang_codes
        ]
    )

    run_summary = {
        "model_id": model_id,
        "languages": lang_codes,
        "budget_key": args.budget_key,
        "steps": global_step,
        "dataset_kind": args.dataset_kind,
        "lambda_chan": args.lambda_chan,
        "dataset_sources": {lc: datasets_by_lang[lc].dataset_source for lc in lang_codes},
        "train_counts": {lc: datasets_by_lang[lc].num_train for lc in lang_codes},
        "eval_counts": {lc: datasets_by_lang[lc].num_eval for lc in lang_codes},
        "final_diag_eval": final_diag_eval,
        "output_dir": str(final_dir),
    }
    write_json(out_dir / "run_summary.json", run_summary)
    write_jsonl(out_dir / "train_log.jsonl", train_log)
    write_jsonl(out_dir / "eval_history.jsonl", eval_history)
    write_json(out_dir / "final_diag_eval.json", final_diag_eval)
    if cross_eval_matrix:
        write_json(out_dir / "final_cross_eval_matrix.json", cross_eval_matrix)

    write_csv(
        out_dir / "train_loss_summary.csv",
        train_loss_summary,
        [
            "lang",
            "steps_seen",
            "mean_loss",
            "first20_mean_loss",
            "last20_mean_loss",
            "final_logged_loss",
            "improvement_pct",
        ],
    )
    write_csv(
        out_dir / "language_budget_summary.csv",
        budget_summary,
        ["lang", "language", "support_layers", "rank_sum", "max_rank", "min_rank"],
    )
    write_csv(
        out_dir / "final_diagonal_eval.csv",
        diagonal_eval_rows,
        ["lang", "language", "own_adapter_ppl", "best_adapter", "best_adapter_ppl", "own_minus_best"],
    )
    if cross_eval_rows:
        write_csv(
            out_dir / "final_cross_eval_matrix.csv",
            cross_eval_rows,
            ["eval_lang", "eval_language"] + lang_codes,
        )
    if eval_history:
        write_csv(
            out_dir / "eval_history.csv",
            eval_history,
            ["step"] + [f"ppl_{lc}" for lc in lang_codes],
        )

    if not args.skip_report_plots:
        render_report_plots(
            out_dir=out_dir,
            train_log=train_log,
            eval_history=eval_history,
            cross_eval_rows=cross_eval_rows,
            diagonal_rows=diagonal_eval_rows,
            language_names=language_names,
        )

    LOGGER.info("Done. Final adapters saved to %s", final_dir)


if __name__ == "__main__":
    main()
