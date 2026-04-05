#!/usr/bin/env python
"""PLoRA Steps 1-4 for task-specific language channels on Llama 3.1 8B."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from plora_task_dataset_utils import prepare_task_data_bundle


LOGGER = logging.getLogger("plora.task.channel_budget")

ATTN_MODULE_HINTS = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP_MODULE_HINTS = ("gate_proj", "up_proj", "down_proj")


class ActivationExtractor:
    """Notebook-aligned hook-based hidden-state extractor."""

    def __init__(self, model):
        self.model = model
        self.activations: Dict[int, torch.Tensor] = {}
        self.hooks = []
        self._register_hooks()

    def _resolve_layers(self):
        core = getattr(self.model, "model", None)
        layers = getattr(core, "layers", None)
        if layers is None:
            raise RuntimeError("Expected decoder layers under model.model.layers")
        return layers

    def _register_hooks(self) -> None:
        for idx, layer in enumerate(self._resolve_layers()):
            self.hooks.append(layer.register_forward_hook(self._build_hook(idx)))

    def _build_hook(self, layer_idx: int):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            self.activations[layer_idx] = hidden.detach()

        return hook

    def extract(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[int, torch.Tensor]:
        self.activations = {}
        with torch.no_grad():
            self.model(input_ids=input_ids, attention_mask=attention_mask)
        return self.activations

    def cleanup(self) -> None:
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Task-specific PLoRA channel analysis and rank budgeting")
    p.add_argument("--dataset-manifest", type=str, required=True)
    p.add_argument("--task", type=str, required=True, choices=["summarization", "qa", "sentiment"])
    p.add_argument("--model-id", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--target-languages", type=str, default="")
    p.add_argument("--probe-samples", type=int, default=1000)
    p.add_argument("--train-samples-for-budget", type=int, default=0)
    p.add_argument("--eval-samples", type=int, default=128)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--subsample-rounds", type=int, default=5)
    p.add_argument("--subsample-size", type=int, default=200)
    p.add_argument("--kmin-frac", type=float, default=0.10)
    p.add_argument("--kmax-frac", type=float, default=0.90)
    p.add_argument("--k-alpha", type=int, default=10)
    p.add_argument("--terminal-window", type=int, default=0, help="0 = auto floor(0.1 * n_layers)")
    p.add_argument("--r-lambda-equal", type=float, default=256.0)
    p.add_argument("--r-min", type=int, default=4)
    p.add_argument("--r-max", type=int, default=64)
    p.add_argument("--p-exp", type=float, default=2.0)
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--r-fair-min", type=float, default=128.0)
    p.add_argument("--r-fair-max", type=float, default=512.0)
    p.add_argument(
        "--fairness-size-mode",
        type=str,
        default="raw",
        choices=["raw", "effective"],
        help="Use raw dataset size or actual prepared train-count for fair-budget scaling.",
    )
    p.add_argument(
        "--target-set",
        type=str,
        default="full",
        choices=["full", "attention_only", "mlp_only"],
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--use-bf16", action="store_true")
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


def extract_activations(
    sentences: List[str],
    tokenizer,
    extractor: ActivationExtractor,
    device: torch.device,
    n_layers: int,
    max_length: int,
) -> Dict[int, torch.Tensor]:
    all_acts = {i: [] for i in range(n_layers)}
    for sent in tqdm(sentences, desc="probe", leave=False):
        inputs = tokenizer(
            sent,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True,
        )
        acts = extractor.extract(
            inputs["input_ids"].to(device),
            inputs["attention_mask"].to(device),
        )
        for layer_idx, act in acts.items():
            mean_act = act.reshape(-1, act.size(-1)).mean(dim=0).cpu()
            all_acts[layer_idx].append(mean_act)
    return {layer_idx: torch.stack(vals) for layer_idx, vals in all_acts.items()}


def compute_power_law_alpha(
    activations: torch.Tensor,
    kmin_frac: float,
    kmax_frac: float,
) -> float:
    centered = activations - activations.mean(dim=0, keepdim=True)
    try:
        if centered.dtype == torch.float16:
            centered = centered.float()
        singular_values = torch.linalg.svdvals(centered).cpu().numpy()
    except Exception:
        return 0.0

    singular_values = singular_values[singular_values > 1e-10]
    rank = len(singular_values)
    if rank < 4:
        return 0.0

    kmin = max(1, int(kmin_frac * rank))
    kmax = min(rank, int(kmax_frac * rank))
    if kmax <= kmin:
        kmax = min(rank, kmin + 1)
    if kmax <= kmin:
        return 0.0

    tail_ranks = np.arange(kmin, kmax + 1)
    tail_sigmas = singular_values[kmin - 1 : kmax]
    log_k = np.log(tail_ranks)
    log_s = np.log(tail_sigmas + 1e-12)
    slope, _ = np.polyfit(log_k, log_s, 1)
    beta = -float(slope)
    if beta < 1e-6:
        return 0.0
    return 1.0 / beta


def compute_alpha_stability(
    activations: torch.Tensor,
    n_subsamples: int,
    subsample_size: int,
    kmin_frac: float,
    kmax_frac: float,
) -> Tuple[float, float]:
    sample_count = activations.shape[0]
    use_size = min(subsample_size, sample_count)
    alphas: List[float] = []
    for _ in range(n_subsamples):
        idx = torch.randperm(sample_count)[:use_size]
        alpha = compute_power_law_alpha(activations[idx], kmin_frac, kmax_frac)
        alphas.append(alpha)
    arr = np.array(alphas, dtype=np.float64)
    mean_a = float(arr.mean()) if len(arr) else 0.0
    std_a = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return mean_a, std_a


def top_k_layers(d_tilde_lang: np.ndarray, k_alpha: int) -> List[int]:
    ranked = np.argsort(d_tilde_lang)[::-1]
    return sorted(int(x) for x in ranked[: min(k_alpha, len(ranked))])


def build_support_set(
    d_tilde_lang: np.ndarray,
    n_layers: int,
    k_alpha: int,
    terminal_window: int,
) -> Tuple[List[int], List[int], List[int], List[int]]:
    topk = set(top_k_layers(d_tilde_lang, k_alpha))
    s_term = set(range(n_layers - terminal_window, n_layers)) if terminal_window > 0 else set()
    overlap = sorted(topk & s_term)
    support = sorted(topk | s_term)
    return support, sorted(topk), sorted(s_term), overlap


def smooth_alpha_curve(alpha_lang: np.ndarray, window: int = 3) -> np.ndarray:
    if window <= 1:
        return alpha_lang.copy()
    pad = window // 2
    padded = np.pad(alpha_lang, (pad, pad), mode="reflect")
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def recompute_dtilde_from_smooth(alpha_mean_all: Dict[str, np.ndarray], lang_codes: List[str]) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    alpha_smooth = {lc: smooth_alpha_curve(alpha_mean_all[lc], window=3) for lc in lang_codes}
    n_layers = len(next(iter(alpha_smooth.values())))
    mu_s = np.array(
        [np.mean([alpha_smooth[lc][ell] for lc in lang_codes]) for ell in range(n_layers)],
        dtype=np.float64,
    )
    sigma_s = np.array(
        [
            math.sqrt(np.mean([(alpha_smooth[lc][ell] - mu_s[ell]) ** 2 for lc in lang_codes]) + 1e-8)
            for ell in range(n_layers)
        ],
        dtype=np.float64,
    )
    d_s = {lc: np.abs(alpha_smooth[lc] - mu_s) / (sigma_s + 1e-12) for lc in lang_codes}
    return d_s, mu_s, sigma_s


def margin_condition(
    d_tilde_lang: np.ndarray,
    sd_lang: np.ndarray,
    sigma_ref: np.ndarray,
    k_alpha: int,
) -> Dict[str, Any]:
    total_layers = len(d_tilde_lang)
    ranked = np.argsort(d_tilde_lang)[::-1]
    k_idx = k_alpha - 1
    k1_idx = k_alpha
    if k1_idx >= total_layers:
        return {
            "margin": float("inf"),
            "epsilon": 0.0,
            "epsilon_local": 0.0,
            "threshold": 0.0,
            "is_stable": True,
            "layer_k": int(ranked[min(k_idx, total_layers - 1)]),
            "layer_k1": None,
            "score_k": float(d_tilde_lang[int(ranked[min(k_idx, total_layers - 1)])]),
            "score_k1": 0.0,
        }

    ell_k = int(ranked[k_idx])
    ell_k1 = int(ranked[k1_idx])
    score_k = float(d_tilde_lang[ell_k])
    score_k1 = float(d_tilde_lang[ell_k1])
    margin = score_k - score_k1
    eps_per_layer = sd_lang / (sigma_ref + 1e-12)
    epsilon = float(np.mean(eps_per_layer))
    epsilon_local = float(eps_per_layer[ell_k])
    threshold = 2.0 * epsilon
    return {
        "margin": float(margin),
        "epsilon": epsilon,
        "epsilon_local": epsilon_local,
        "threshold": float(threshold),
        "is_stable": bool(margin > threshold),
        "layer_k": ell_k,
        "layer_k1": ell_k1,
        "score_k": score_k,
        "score_k1": score_k1,
    }


def find_max_stable_k(
    d_tilde_lang: np.ndarray,
    sd_lang: np.ndarray,
    sigma_ref: np.ndarray,
    k_max: int,
    k_min: int = 4,
) -> int:
    for candidate in range(k_max, k_min - 1, -1):
        if margin_condition(d_tilde_lang, sd_lang, sigma_ref, candidate)["is_stable"]:
            return candidate
    return k_min


def allocate_ranks(
    d_tilde_lang: List[float],
    support_set: List[int],
    budget: float,
    r_min: int,
    r_max: int,
    p_exp: float,
) -> Dict[int, int]:
    if not support_set:
        return {}
    weights = {ell: max(float(d_tilde_lang[ell]), 1e-8) ** p_exp for ell in support_set}
    normalizer = sum(weights.values()) + 1e-12
    ranks = {}
    for ell in support_set:
        raw = budget * weights[ell] / normalizer
        ranks[ell] = int(np.clip(round(raw), r_min, r_max))
    return ranks


def compute_fair_budgets(
    dataset_sizes: Dict[str, int],
    lang_codes: List[str],
    base_budget: float,
    gamma: float,
    r_min: float,
    r_max: float,
) -> Dict[str, float]:
    raw = {lc: max(dataset_sizes.get(lc, 1), 1) ** (-gamma) for lc in lang_codes}
    median_raw = float(np.median(list(raw.values())))
    scale = base_budget / (median_raw + 1e-12)
    return {lc: float(np.clip(raw[lc] * scale, r_min, r_max)) for lc in lang_codes}


def infer_model_arch(config) -> Dict[str, int]:
    hidden_size = int(config.hidden_size)
    num_attention_heads = int(config.num_attention_heads)
    num_key_value_heads = int(getattr(config, "num_key_value_heads", num_attention_heads))
    head_dim = int(getattr(config, "head_dim", hidden_size // num_attention_heads))
    intermediate_size = int(config.intermediate_size)
    return {
        "hidden_size": hidden_size,
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_key_value_heads,
        "head_dim": head_dim,
        "intermediate_size": intermediate_size,
        "num_hidden_layers": int(config.num_hidden_layers),
        "vocab_size": int(config.vocab_size),
    }


def build_target_shapes(model_arch: Dict[str, int], target_set: str) -> Dict[str, Tuple[int, int]]:
    hidden_size = model_arch["hidden_size"]
    kv_dim = model_arch["num_key_value_heads"] * model_arch["head_dim"]
    intermediate = model_arch["intermediate_size"]

    full = {
        "q_proj": (hidden_size, hidden_size),
        "k_proj": (hidden_size, kv_dim),
        "v_proj": (hidden_size, kv_dim),
        "o_proj": (hidden_size, hidden_size),
        "gate_proj": (hidden_size, intermediate),
        "up_proj": (hidden_size, intermediate),
        "down_proj": (intermediate, hidden_size),
    }
    if target_set == "attention_only":
        return {k: v for k, v in full.items() if k in ATTN_MODULE_HINTS}
    if target_set == "mlp_only":
        return {k: v for k, v in full.items() if k in MLP_MODULE_HINTS}
    return full


def lora_param_cost_per_layer(rank: int, target_shapes: Dict[str, Tuple[int, int]]) -> int:
    return int(sum(rank * (din + dout) for din, dout in target_shapes.values()))


def lora_param_cost_language(rank_map: Dict[int, int], target_shapes: Dict[str, Tuple[int, int]]) -> int:
    return int(sum(lora_param_cost_per_layer(rank, target_shapes) for rank in rank_map.values()))


def serialise_rank_map(all_ranks: Dict[str, Dict[int, int]]) -> Dict[str, Dict[str, int]]:
    return {lang: {str(layer): int(rank) for layer, rank in rank_map.items()} for lang, rank_map in all_ranks.items()}


def determine_terminal_window(n_layers: int, requested: int) -> int:
    if requested > 0:
        return requested
    return max(1, int(math.floor(0.1 * n_layers)))


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.fairness_size_mode == "effective" and args.train_samples_for_budget <= 0:
        raise ValueError("--train-samples-for-budget must be > 0 when --fairness-size-mode=effective")

    target_languages = [x.strip() for x in args.target_languages.split(",") if x.strip()]
    bundle = prepare_task_data_bundle(
        manifest_path=args.dataset_manifest,
        task=args.task,
        target_languages=target_languages or None,
        probe_limit=args.probe_samples,
        train_limit=max(1, args.train_samples_for_budget),
        eval_limit=args.eval_samples,
        seed=args.seed,
        build_train_records=False,
        build_eval_records=False,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lang_codes = sorted(bundle["by_lang"])
    language_names = bundle["language_names"]

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if (args.use_bf16 and torch.cuda.is_available()) else torch.float16
    if not torch.cuda.is_available():
        dtype = torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
    )
    if not torch.cuda.is_available():
        model.to(device)
    model.eval()

    extractor = ActivationExtractor(model)
    n_layers = int(model.config.num_hidden_layers)
    terminal_window = determine_terminal_window(n_layers, args.terminal_window)

    probe_count = min(len(bundle["by_lang"][lc]["probe_texts"]) for lc in lang_codes)
    if probe_count < 8:
        raise RuntimeError(f"Need at least 8 probe texts per language; found minimum {probe_count}")
    LOGGER.info("Using %d equalized probe texts per language for task=%s", probe_count, args.task)

    language_activations: Dict[str, Dict[int, torch.Tensor]] = {}
    for lc in lang_codes:
        probe_texts = bundle["by_lang"][lc]["probe_texts"][:probe_count]
        LOGGER.info("Extracting layer activations for %s (%s)", language_names.get(lc, lc), lc)
        language_activations[lc] = extract_activations(
            sentences=probe_texts,
            tokenizer=tokenizer,
            extractor=extractor,
            device=device,
            n_layers=n_layers,
            max_length=args.max_length,
        )

    extractor.cleanup()

    alpha_mean: Dict[str, np.ndarray] = {}
    alpha_std: Dict[str, np.ndarray] = {}
    for lc in lang_codes:
        means, stds = [], []
        for layer_idx in range(n_layers):
            mean_a, std_a = compute_alpha_stability(
                activations=language_activations[lc][layer_idx],
                n_subsamples=args.subsample_rounds,
                subsample_size=args.subsample_size,
                kmin_frac=args.kmin_frac,
                kmax_frac=args.kmax_frac,
            )
            means.append(mean_a)
            stds.append(std_a)
        alpha_mean[lc] = np.array(means, dtype=np.float64)
        alpha_std[lc] = np.array(stds, dtype=np.float64)

    mu = np.array([np.mean([alpha_mean[lc][ell] for lc in lang_codes]) for ell in range(n_layers)], dtype=np.float64)
    sigma = np.array(
        [
            math.sqrt(np.mean([(alpha_mean[lc][ell] - mu[ell]) ** 2 for lc in lang_codes]) + 1e-8)
            for ell in range(n_layers)
        ],
        dtype=np.float64,
    )
    d_tilde = {lc: np.abs(alpha_mean[lc] - mu) / (sigma + 1e-12) for lc in lang_codes}

    support_sets_initial: Dict[str, List[int]] = {}
    support_alpha: Dict[str, List[int]] = {}
    support_term: Dict[str, List[int]] = {}
    support_overlap: Dict[str, List[int]] = {}
    for lc in lang_codes:
        support, topk_only, term_only, overlap = build_support_set(
            d_tilde_lang=d_tilde[lc],
            n_layers=n_layers,
            k_alpha=args.k_alpha,
            terminal_window=terminal_window,
        )
        support_sets_initial[lc] = support
        support_alpha[lc] = topk_only
        support_term[lc] = term_only
        support_overlap[lc] = overlap

    stability = {lc: margin_condition(d_tilde[lc], alpha_std[lc], sigma, args.k_alpha) for lc in lang_codes}
    d_tilde_smooth, mu_smooth, sigma_smooth = recompute_dtilde_from_smooth(alpha_mean, lang_codes)

    support_sets_final: Dict[str, List[int]] = {}
    support_strategy: Dict[str, str] = {}
    effective_k: Dict[str, int] = {}
    for lc in lang_codes:
        if stability[lc]["is_stable"]:
            support_sets_final[lc] = support_sets_initial[lc]
            support_strategy[lc] = "original"
            effective_k[lc] = args.k_alpha
            continue

        smooth_margin = margin_condition(d_tilde_smooth[lc], alpha_std[lc], sigma_smooth, args.k_alpha)
        if smooth_margin["is_stable"]:
            support_sets_final[lc] = build_support_set(
                d_tilde_lang=d_tilde_smooth[lc],
                n_layers=n_layers,
                k_alpha=args.k_alpha,
                terminal_window=terminal_window,
            )[0]
            support_strategy[lc] = "smoothed(w=3)"
            effective_k[lc] = args.k_alpha
            continue

        fallback_k = find_max_stable_k(d_tilde[lc], alpha_std[lc], sigma, args.k_alpha)
        support_sets_final[lc] = build_support_set(
            d_tilde_lang=d_tilde[lc],
            n_layers=n_layers,
            k_alpha=fallback_k,
            terminal_window=terminal_window,
        )[0]
        support_strategy[lc] = f"smaller_K(K={fallback_k})"
        effective_k[lc] = fallback_k

    model_arch = infer_model_arch(model.config)
    target_shapes = build_target_shapes(model_arch, args.target_set)

    dataset_sizes: Dict[str, int] = {}
    for lc in lang_codes:
        if args.fairness_size_mode == "effective":
            dataset_sizes[lc] = int(bundle["by_lang"][lc]["effective_train_count"])
        else:
            dataset_sizes[lc] = int(bundle["by_lang"][lc]["raw_train_count"])

    equal_budgets = {lc: float(args.r_lambda_equal) for lc in lang_codes}
    fair_budgets = compute_fair_budgets(
        dataset_sizes=dataset_sizes,
        lang_codes=lang_codes,
        base_budget=args.r_lambda_equal,
        gamma=args.gamma,
        r_min=args.r_fair_min,
        r_max=args.r_fair_max,
    )

    all_ranks_equal = {
        lc: allocate_ranks(
            d_tilde_lang=d_tilde[lc].tolist(),
            support_set=support_sets_final[lc],
            budget=equal_budgets[lc],
            r_min=args.r_min,
            r_max=args.r_max,
            p_exp=args.p_exp,
        )
        for lc in lang_codes
    }
    all_ranks_fair = {
        lc: allocate_ranks(
            d_tilde_lang=d_tilde[lc].tolist(),
            support_set=support_sets_final[lc],
            budget=fair_budgets[lc],
            r_min=args.r_min,
            r_max=args.r_max,
            p_exp=args.p_exp,
        )
        for lc in lang_codes
    }

    selection_frequency = np.zeros(n_layers, dtype=int)
    for lc in lang_codes:
        for ell in support_sets_final[lc]:
            selection_frequency[ell] += 1
    all_sets = [set(support_sets_final[lc]) for lc in lang_codes]
    universal_layers = sorted(all_sets[0].intersection(*all_sets[1:])) if all_sets else []
    majority_threshold = max(1, int(math.ceil(0.75 * len(lang_codes))))
    majority_layers = [ell for ell in range(n_layers) if int(selection_frequency[ell]) >= majority_threshold]

    rows_csv: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    for lc in lang_codes:
        rm_eq = all_ranks_equal[lc]
        rm_fair = all_ranks_fair[lc]
        rs_eq = list(rm_eq.values()) or [0]
        rs_fair = list(rm_fair.values()) or [0]
        summary_rows.append(
            {
                "Language": language_names.get(lc, lc),
                "Code": lc,
                "Task": args.task,
                "ProbeCount": probe_count,
                "SupportStrategy": support_strategy[lc],
                "EffectiveK": effective_k[lc],
                "SupportSize": len(support_sets_final[lc]),
                "RawTrainSize": bundle["by_lang"][lc]["raw_train_count"],
                "EffectiveTrainSize": bundle["by_lang"][lc]["effective_train_count"],
                "EqualRankSum": sum(rs_eq),
                "FairBudget": round(fair_budgets[lc], 3),
                "FairRankSum": sum(rs_fair),
                "EqualParamsM": round(lora_param_cost_language(rm_eq, target_shapes) / 1e6, 6),
                "FairParamsM": round(lora_param_cost_language(rm_fair, target_shapes) / 1e6, 6),
                "Stable": int(stability[lc]["is_stable"]),
                "Margin": round(stability[lc]["margin"], 6),
                "Threshold2Eps": round(stability[lc]["threshold"], 6),
                "MaxDTildeLayer": int(np.argmax(d_tilde[lc])),
                "MaxDTildeValue": round(float(np.max(d_tilde[lc])), 6),
            }
        )
        for ell in range(n_layers):
            rows_csv.append(
                {
                    "lang_code": lc,
                    "lang_name": language_names.get(lc, lc),
                    "task": args.task,
                    "layer": ell,
                    "alpha_mean": round(float(alpha_mean[lc][ell]), 6),
                    "alpha_std": round(float(alpha_std[lc][ell]), 6),
                    "d_tilde": round(float(d_tilde[lc][ell]), 6),
                    "in_support": int(ell in set(support_sets_final[lc])),
                    "in_topk": int(ell in set(support_alpha[lc])),
                    "in_terminal_window": int(ell in set(support_term[lc])),
                    "selection_frequency": int(selection_frequency[ell]),
                    "rank_equal": int(all_ranks_equal[lc].get(ell, 0)),
                    "rank_fair": int(all_ranks_fair[lc].get(ell, 0)),
                }
            )

    payload = {
        "metadata": {
            "task": args.task,
            "dataset_manifest": str(Path(args.dataset_manifest)),
            "model_id": args.model_id,
            "probe_samples_requested": args.probe_samples,
            "probe_samples_effective": probe_count,
            "subsample_rounds": args.subsample_rounds,
            "subsample_size": args.subsample_size,
            "kmin_frac": args.kmin_frac,
            "kmax_frac": args.kmax_frac,
            "n_layers": n_layers,
            "n_languages": len(lang_codes),
            "languages": {lc: language_names.get(lc, lc) for lc in lang_codes},
            "R_lambda_equal": args.r_lambda_equal,
            "R_min": args.r_min,
            "R_max": args.r_max,
            "p_exp": args.p_exp,
            "gamma": args.gamma,
            "R_fair_min": args.r_fair_min,
            "R_fair_max": args.r_fair_max,
            "fair_budgets": {lc: round(v, 6) for lc, v in fair_budgets.items()},
            "K_alpha": args.k_alpha,
            "terminal_window": terminal_window,
            "step3_source": "script_computed",
            "model_arch": model_arch,
            "lora_target_set": args.target_set,
            "lora_targets": {k: list(v) for k, v in target_shapes.items()},
            "total_params_equal_M": round(
                sum(lora_param_cost_language(all_ranks_equal[lc], target_shapes) for lc in lang_codes) / 1e6,
                6,
            ),
            "total_params_fair_M": round(
                sum(lora_param_cost_language(all_ranks_fair[lc], target_shapes) for lc in lang_codes) / 1e6,
                6,
            ),
        },
        "support_sets": {lc: support_sets_final[lc] for lc in lang_codes},
        "equal_budget": serialise_rank_map(all_ranks_equal),
        "fair_budget": serialise_rank_map(all_ranks_fair),
        "channel_metrics": {
            "alpha_mean": {lc: [round(float(x), 6) for x in alpha_mean[lc].tolist()] for lc in lang_codes},
            "alpha_std": {lc: [round(float(x), 6) for x in alpha_std[lc].tolist()] for lc in lang_codes},
            "d_tilde": {lc: [round(float(x), 6) for x in d_tilde[lc].tolist()] for lc in lang_codes},
            "mu": [round(float(x), 6) for x in mu.tolist()],
            "sigma": [round(float(x), 6) for x in sigma.tolist()],
        },
        "step3": {
            "support_sets_initial": {lc: support_sets_initial[lc] for lc in lang_codes},
            "support_alpha": {lc: support_alpha[lc] for lc in lang_codes},
            "support_term": {lc: support_term[lc] for lc in lang_codes},
            "support_overlap": {lc: support_overlap[lc] for lc in lang_codes},
            "support_strategy": support_strategy,
            "effective_k": effective_k,
            "stability": {
                lc: {
                    "margin": round(float(stability[lc]["margin"]), 6),
                    "epsilon": round(float(stability[lc]["epsilon"]), 6),
                    "epsilon_local": round(float(stability[lc]["epsilon_local"]), 6),
                    "threshold_2eps": round(float(stability[lc]["threshold"]), 6),
                    "is_stable": bool(stability[lc]["is_stable"]),
                    "layer_K": stability[lc]["layer_k"],
                    "layer_K_plus_1": stability[lc]["layer_k1"],
                    "score_K": round(float(stability[lc]["score_k"]), 6),
                    "score_K_plus_1": round(float(stability[lc]["score_k1"]), 6),
                }
                for lc in lang_codes
            },
        },
        "cross_language": {
            "universal_layers": universal_layers,
            "majority_layers": majority_layers,
            "selection_frequency": selection_frequency.tolist(),
        },
        "dataset_sizes": dataset_sizes,
        "dataset_sources": {lc: bundle["by_lang"][lc]["dataset_source"] for lc in lang_codes},
    }

    write_json(out_dir / f"{args.task}_rank_budgets.json", payload)
    write_csv(
        out_dir / f"{args.task}_rank_budgets.csv",
        rows_csv,
        [
            "lang_code",
            "lang_name",
            "task",
            "layer",
            "alpha_mean",
            "alpha_std",
            "d_tilde",
            "in_support",
            "in_topk",
            "in_terminal_window",
            "selection_frequency",
            "rank_equal",
            "rank_fair",
        ],
    )
    write_csv(
        out_dir / f"{args.task}_param_summary.csv",
        summary_rows,
        [
            "Language",
            "Code",
            "Task",
            "ProbeCount",
            "SupportStrategy",
            "EffectiveK",
            "SupportSize",
            "RawTrainSize",
            "EffectiveTrainSize",
            "EqualRankSum",
            "FairBudget",
            "FairRankSum",
            "EqualParamsM",
            "FairParamsM",
            "Stable",
            "Margin",
            "Threshold2Eps",
            "MaxDTildeLayer",
            "MaxDTildeValue",
        ],
    )
    LOGGER.info("Wrote task-specific rank budget artifacts to %s", out_dir)


if __name__ == "__main__":
    main()
