#!/usr/bin/env python
"""PLoRA Step 5 routed training for task-specific multilingual adapters."""

from __future__ import annotations

import argparse
import math
import random
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from plora_step5_language_routed_training import (
    build_adapter_module_map,
    cross_eval_to_rows,
    freeze_backbone,
    load_rank_payload,
    set_trainable_adapter,
    stable_rank_penalty,
    summarize_cross_eval,
    summarize_train_log,
    write_csv,
    write_json,
    write_jsonl,
)
from plora_task_dataset_utils import prepare_task_data_bundle


@dataclass
class TaskLangData:
    train_loader: DataLoader
    eval_loader: DataLoader
    num_train: int
    num_eval: int
    dataset_source: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Task-specific PLoRA Step 5 routed training")
    p.add_argument("--dataset-manifest", type=str, required=True)
    p.add_argument("--task", type=str, required=True, choices=["summarization", "qa", "sentiment"])
    p.add_argument("--rank-json", type=str, required=True)
    p.add_argument("--budget-key", type=str, default="fair_budget", choices=["fair_budget", "equal_budget"])
    p.add_argument("--model-id", type=str, default="")
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--target-languages", type=str, default="")
    p.add_argument("--max-train-samples", type=int, default=2000)
    p.add_argument("--max-eval-samples", type=int, default=256)
    p.add_argument("--train-batch-size", type=int, default=2)
    p.add_argument("--eval-batch-size", type=int, default=2)
    p.add_argument("--grad-accum-steps", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--max-length", type=int, default=768)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--lambda-chan", type=float, default=0.0)
    p.add_argument("--power-iters", type=int, default=3)
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--use-bf16", action="store_true")
    p.add_argument("--skip-cross-eval", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def ensure_target_spacing(prompt: str, target: str) -> str:
    if not prompt:
        return target
    if not target:
        return target
    if prompt.endswith((" ", "\n", "\t")) or target.startswith((" ", "\n", "\t")):
        return target
    return " " + target


def encode_supervised_example(tokenizer, prompt: str, target: str, max_length: int) -> Dict[str, List[int]] | None:
    adjusted_target = ensure_target_spacing(prompt, target)
    bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    eos = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []

    prompt_ids = bos + tokenizer(prompt, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(adjusted_target, add_special_tokens=False)["input_ids"] + eos
    if not target_ids:
        return None

    if len(target_ids) >= max_length:
        target_ids = target_ids[: max_length - 1]
        if eos:
            target_ids[-1] = eos[0]

    max_prompt_tokens = max_length - len(target_ids)
    if max_prompt_tokens <= 0:
        max_prompt_tokens = 1 if bos else 0
        if max_prompt_tokens == 0:
            return None

    prompt_ids = prompt_ids[:max_prompt_tokens]
    input_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + target_ids
    attention_mask = [1] * len(input_ids)
    if not any(x != -100 for x in labels):
        return None
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def tokenize_records(records: List[Dict[str, str]], tokenizer, max_length: int) -> Dataset:
    features: List[Dict[str, List[int]]] = []
    for record in records:
        encoded = encode_supervised_example(
            tokenizer=tokenizer,
            prompt=record["prompt"],
            target=record["target"],
            max_length=max_length,
        )
        if encoded is not None:
            features.append(encoded)
    if not features:
        raise RuntimeError("No valid supervised examples remained after tokenization.")
    return Dataset.from_list(features)


def make_collator(tokenizer):
    pad_id = tokenizer.pad_token_id

    def collate(features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, attention_mask, labels = [], [], []
        for feat in features:
            ids = list(feat["input_ids"])
            mask = list(feat["attention_mask"])
            labs = list(feat["labels"])
            pad_len = max_len - len(ids)
            input_ids.append(ids + [pad_id] * pad_len)
            attention_mask.append(mask + [0] * pad_len)
            labels.append(labs + [-100] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    return collate


def build_dataloaders(
    tokenizer,
    bundle: Dict[str, Any],
    lang_codes: List[str],
    train_batch_size: int,
    eval_batch_size: int,
    max_length: int,
) -> Dict[str, TaskLangData]:
    collator = make_collator(tokenizer)
    out: Dict[str, TaskLangData] = {}
    for lc in lang_codes:
        lang_data = bundle["by_lang"][lc]
        train_ds = tokenize_records(lang_data["train_records"], tokenizer, max_length)
        eval_ds = tokenize_records(lang_data["eval_records"], tokenizer, max_length)
        out[lc] = TaskLangData(
            train_loader=DataLoader(train_ds, batch_size=train_batch_size, shuffle=True, collate_fn=collator),
            eval_loader=DataLoader(eval_ds, batch_size=eval_batch_size, shuffle=False, collate_fn=collator),
            num_train=len(train_ds),
            num_eval=len(eval_ds),
            dataset_source=lang_data["dataset_source"],
        )
    return out


def evaluate_masked_ppl(model, loader: DataLoader, device: torch.device, adapter_name: str | None = None) -> float:
    model.eval()
    if adapter_name is None:
        adapter_ctx = model.disable_adapter() if hasattr(model, "disable_adapter") else nullcontext()
    else:
        model.set_adapter(adapter_name)
        adapter_ctx = nullcontext()
    losses: List[float] = []
    with adapter_ctx:
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(**batch)
                losses.append(float(out.loss.detach().float().item()))
    if not losses:
        return float("inf")
    return float(math.exp(sum(losses) / len(losses)))


def evaluate_ppl_matrix(
    model,
    datasets_by_lang: Dict[str, TaskLangData],
    device: torch.device,
    adapter_langs: List[str],
    eval_langs: List[str],
) -> Dict[str, Dict[str, float]]:
    matrix: Dict[str, Dict[str, float]] = {}
    for eval_lang in eval_langs:
        loader = datasets_by_lang[eval_lang].eval_loader
        matrix[eval_lang] = {adapter_lang: evaluate_masked_ppl(model, loader, device, adapter_lang) for adapter_lang in adapter_langs}
    return matrix


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    random.seed(args.seed)

    payload = load_rank_payload(args.rank_json)
    model_id = args.model_id or payload["metadata"].get("model_id", "meta-llama/Llama-3.1-8B-Instruct")
    support_sets: Dict[str, List[int]] = payload["support_sets"]
    rank_map_raw: Dict[str, Dict[str, int]] = payload[args.budget_key]
    language_names: Dict[str, str] = dict(payload["metadata"].get("languages", {}))

    target_languages = [x.strip() for x in args.target_languages.split(",") if x.strip()]
    bundle = prepare_task_data_bundle(
        manifest_path=args.dataset_manifest,
        task=args.task,
        target_languages=target_languages or None,
        probe_limit=1,
        train_limit=args.max_train_samples,
        eval_limit=args.max_eval_samples,
        seed=args.seed,
        build_train_records=True,
        build_eval_records=True,
    )
    language_names.update(bundle["language_names"])

    lang_codes = sorted(set(bundle["by_lang"]) & set(support_sets) & set(rank_map_raw))
    if not lang_codes:
        raise RuntimeError("No overlapping languages between manifest and rank json.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    datasets_by_lang = build_dataloaders(
        tokenizer=tokenizer,
        bundle=bundle,
        lang_codes=lang_codes,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        max_length=args.max_length,
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
        low_cpu_mem_usage=True,
    )
    if not torch.cuda.is_available():
        model.to(device)

    base_module_names = [name for name, _ in model.named_modules()]
    target_hints = payload["metadata"].get("lora_targets", {})
    target_names = sorted(target_hints.keys()) if isinstance(target_hints, dict) and target_hints else sorted(
        {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    )

    peft_model: PeftModel | None = None
    for idx, lc in enumerate(lang_codes):
        support = [int(x) for x in support_sets[lc]]
        rank_int = {int(k): int(v) for k, v in rank_map_raw[lc].items() if int(v) > 0}
        rank_pattern_template = build_adapter_module_map(base_module_names, support, target_names)
        rank_pattern: Dict[str, int] = {}
        alpha_pattern: Dict[str, int] = {}
        for module_name in rank_pattern_template:
            layer_match = module_name.split(".layers.")
            if len(layer_match) < 2:
                continue
            layer_idx = int(layer_match[1].split(".", 1)[0])
            rank = int(rank_int.get(layer_idx, 0))
            if rank > 0:
                rank_pattern[module_name] = rank
                alpha_pattern[module_name] = max(8, 2 * rank)
        if not rank_pattern:
            raise RuntimeError(f"No LoRA target modules resolved for {lc}")

        cfg_kwargs = {
            "task_type": TaskType.CAUSAL_LM,
            "target_modules": sorted(rank_pattern.keys()),
            "r": max(rank_pattern.values()),
            "lora_alpha": max(alpha_pattern.values()),
            "rank_pattern": rank_pattern,
            "alpha_pattern": alpha_pattern,
            "lora_dropout": 0.05,
            "bias": "none",
        }
        cfg = LoraConfig(**cfg_kwargs)
        if idx == 0:
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
    train_log: List[Dict[str, Any]] = []
    eval_history: List[Dict[str, Any]] = []
    target_steps = 2 if args.dry_run else args.max_steps
    global_step = 0

    pbar = tqdm(total=target_steps, desc=f"{args.task}-train", ncols=100)
    while global_step < target_steps:
        for lc in lang_codes:
            if global_step >= target_steps:
                break
            global_step += 1
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
                target_ranks = {int(k): int(v) for k, v in rank_map_raw[lc].items()}
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
                avg = sum(row["loss"] for row in recent) / len(recent)
                print(f"step={global_step} lang={lc} avg_loss={avg:.4f}")

            if (not args.dry_run) and args.eval_every > 0 and (global_step % args.eval_every == 0):
                row: Dict[str, Any] = {"step": global_step}
                for elc in lang_codes:
                    row[f"base_ppl_{elc}"] = evaluate_masked_ppl(model, datasets_by_lang[elc].eval_loader, device, None)
                    row[f"adapter_ppl_{elc}"] = evaluate_masked_ppl(model, datasets_by_lang[elc].eval_loader, device, elc)
                eval_history.append(row)

            if (not args.dry_run) and args.save_every > 0 and (global_step % args.save_every == 0):
                ckpt = out_dir / f"checkpoint_step_{global_step}"
                ckpt.mkdir(parents=True, exist_ok=True)
                for slc in lang_codes:
                    model.set_adapter(slc)
                    model.save_pretrained(ckpt / f"adapter_{slc}", selected_adapters=[slc])
                tokenizer.save_pretrained(ckpt / "tokenizer")

            pbar.update(1)

    pbar.close()

    for opt in optimizers.values():
        should_step = False
        for group in opt.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    should_step = True
                    break
            if should_step:
                break
        if should_step:
            opt.step()
            opt.zero_grad(set_to_none=True)

    final_dir = out_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    for lc in lang_codes:
        model.set_adapter(lc)
        model.save_pretrained(final_dir / f"adapter_{lc}", selected_adapters=[lc])
    tokenizer.save_pretrained(final_dir / "tokenizer")

    base_diag_eval: Dict[str, float] = {}
    own_adapter_eval: Dict[str, float] = {}
    for lc in lang_codes:
        loader = datasets_by_lang[lc].eval_loader
        base_diag_eval[lc] = evaluate_masked_ppl(model, loader, device, None)
        own_adapter_eval[lc] = evaluate_masked_ppl(model, loader, device, lc)

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
    cross_eval_rows = cross_eval_to_rows(language_names, cross_eval_matrix, lang_codes) if cross_eval_matrix else []
    diagonal_eval_rows = summarize_cross_eval(language_names, cross_eval_matrix) if cross_eval_matrix else []
    if not diagonal_eval_rows:
        diagonal_eval_rows = []
        for lc in lang_codes:
            diagonal_eval_rows.append(
                {
                    "lang": lc,
                    "language": language_names.get(lc, lc),
                    "own_adapter_ppl": own_adapter_eval[lc],
                    "best_adapter": lc,
                    "best_adapter_ppl": own_adapter_eval[lc],
                    "own_minus_best": 0.0,
                }
            )

    task_eval_rows: List[Dict[str, Any]] = []
    for row in diagonal_eval_rows:
        lc = row["lang"]
        base_ppl = float(base_diag_eval[lc])
        adapter_ppl = float(own_adapter_eval[lc])
        change = adapter_ppl - base_ppl
        pct_change = 100.0 * change / base_ppl if base_ppl > 0 else 0.0
        task_eval_rows.append(
            {
                "task": args.task,
                "lang": lc,
                "language": language_names.get(lc, lc),
                "base_ppl": base_ppl,
                "own_adapter_ppl": adapter_ppl,
                "ppl_change": change,
                "ppl_change_pct": pct_change,
                "direction": "decrease" if change < 0 else "increase" if change > 0 else "flat",
                "best_adapter": row["best_adapter"],
                "best_adapter_ppl": row["best_adapter_ppl"],
                "own_minus_best": row["own_minus_best"],
            }
        )

    run_summary = {
        "task": args.task,
        "model_id": model_id,
        "dataset_manifest": str(Path(args.dataset_manifest)),
        "rank_json": str(Path(args.rank_json)),
        "languages": lang_codes,
        "budget_key": args.budget_key,
        "steps": global_step,
        "lambda_chan": args.lambda_chan,
        "dataset_sources": {lc: datasets_by_lang[lc].dataset_source for lc in lang_codes},
        "train_counts": {lc: datasets_by_lang[lc].num_train for lc in lang_codes},
        "eval_counts": {lc: datasets_by_lang[lc].num_eval for lc in lang_codes},
        "base_diag_eval": base_diag_eval,
        "own_adapter_eval": own_adapter_eval,
        "output_dir": str(final_dir),
    }

    write_json(out_dir / "run_summary.json", run_summary)
    write_jsonl(out_dir / "train_log.jsonl", train_log)
    write_jsonl(out_dir / "eval_history.jsonl", eval_history)
    write_json(out_dir / "base_diag_eval.json", base_diag_eval)
    write_json(out_dir / "own_adapter_eval.json", own_adapter_eval)
    if cross_eval_matrix:
        write_json(out_dir / "final_cross_eval_matrix.json", cross_eval_matrix)

    write_csv(
        out_dir / "train_loss_summary.csv",
        train_loss_summary,
        ["lang", "steps_seen", "mean_loss", "first20_mean_loss", "last20_mean_loss", "final_logged_loss", "improvement_pct"],
    )
    write_csv(
        out_dir / "final_diagonal_eval.csv",
        diagonal_eval_rows,
        ["lang", "language", "own_adapter_ppl", "best_adapter", "best_adapter_ppl", "own_minus_best"],
    )
    write_csv(
        out_dir / "final_task_eval.csv",
        task_eval_rows,
        [
            "task",
            "lang",
            "language",
            "base_ppl",
            "own_adapter_ppl",
            "ppl_change",
            "ppl_change_pct",
            "direction",
            "best_adapter",
            "best_adapter_ppl",
            "own_minus_best",
        ],
    )
    if cross_eval_rows:
        write_csv(out_dir / "final_cross_eval_matrix.csv", cross_eval_rows, ["eval_lang", "eval_language"] + lang_codes)
    if eval_history:
        eval_columns = ["step"] + [f"base_ppl_{lc}" for lc in lang_codes] + [f"adapter_ppl_{lc}" for lc in lang_codes]
        write_csv(out_dir / "eval_history.csv", eval_history, eval_columns)


if __name__ == "__main__":
    main()
