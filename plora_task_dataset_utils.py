#!/usr/bin/env python
"""Task dataset utilities for PLoRA task-specific channel analysis and training.

The existing repository already defines the language-channel, support-set, and
rank-budget methods. This module adds only the missing task-data plumbing needed
to run those methods on summarization, QA, and sentiment datasets.
"""

from __future__ import annotations

import json
import math
import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from datasets import Dataset, load_dataset


SUPPORTED_TASKS = {"summarization", "qa", "sentiment"}

DEFAULT_LANGUAGE_NAMES: Dict[str, str] = {
    "eng_Latn": "English",
    "fra_Latn": "French",
    "cmn_Hans": "Chinese",
    "nld_Latn": "Dutch",
    "pol_Latn": "Polish",
    "hin_Deva": "Hindi",
    "ben_Beng": "Bengali",
    "mar_Deva": "Marathi",
    "urd_Arab": "Urdu",
}


@dataclass(frozen=True)
class TaskLanguageSpec:
    builder: str | None
    dataset: str
    config: str | None
    data_files: Dict[str, Any] | None
    builder_kwargs: Dict[str, Any] | None
    expand_mode: str | None
    train_split: str
    eval_split: str
    test_split: str | None
    language_name: str
    prompt_template: str | None
    probe_template: str | None
    target_template: str | None
    system_prompt: str | None
    instruction: str | None
    input_field: str | None
    target_field: str | None
    context_field: str | None
    question_field: str | None
    choices_field: str | None
    answer_field: str | None
    answer_index_field: str | None
    label_field: str | None
    label_map: Dict[str, str]

    @property
    def source_label(self) -> str:
        if self.builder:
            return f"{self.builder}::{self.dataset}"
        if self.config:
            return f"{self.dataset}::{self.config}"
        return self.dataset


def load_task_manifest(path: str | Path) -> Dict[str, Any]:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if "tasks" not in payload or not isinstance(payload["tasks"], dict):
        raise ValueError("Dataset manifest must contain a top-level 'tasks' object.")
    return payload


def _coalesce(value: Any, default: Any) -> Any:
    return default if value is None else value


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, dict):
        if "text" in value:
            return _stringify(value["text"])
        if "value" in value:
            return _stringify(value["value"])
        joined = []
        for key in ("answer", "answers", "label", "labels"):
            if key in value:
                txt = _stringify(value[key])
                if txt:
                    joined.append(txt)
        if joined:
            return " ".join(joined).strip()
        parts = [_stringify(v) for v in value.values()]
        return " ".join([p for p in parts if p]).strip()
    if isinstance(value, (list, tuple)):
        parts = [_stringify(v) for v in value]
        return " ".join([p for p in parts if p]).strip()
    return str(value).strip()


def _extract_field(example: Dict[str, Any], field_name: str | None) -> Any:
    if not field_name:
        return None
    if field_name in example:
        return example[field_name]

    current: Any = example
    for part in field_name.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, (list, tuple)):
            next_values = []
            for item in current:
                if isinstance(item, dict) and part in item:
                    next_values.append(item[part])
            if not next_values:
                return None
            current = next_values
        else:
            return None
    return current


def _normalize_choice_list(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, dict):
        if "text" in raw:
            raw = raw["text"]
        elif "choices" in raw:
            raw = raw["choices"]
        else:
            raw = list(raw.values())
    if isinstance(raw, (list, tuple)):
        out = []
        for item in raw:
            txt = _stringify(item)
            if txt:
                out.append(txt)
        return out
    txt = _stringify(raw)
    return [txt] if txt else []


def _normalize_data_files(raw: Any) -> Dict[str, Any] | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("data_files must be an object mapping split names to file paths.")
    return dict(raw)


def _normalize_mapping(raw: Any) -> Dict[str, Any] | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("Expected an object.")
    return dict(raw)


def _resolve_answer_text(example: Dict[str, Any], spec: TaskLanguageSpec, choices: List[str]) -> str:
    answer_value = _extract_field(example, spec.answer_field)
    if answer_value is not None:
        if isinstance(answer_value, str):
            stripped = answer_value.strip()
            if stripped.startswith(("[", "{", "(")):
                try:
                    answer_value = ast.literal_eval(stripped)
                except Exception:
                    pass
        if isinstance(answer_value, dict):
            if "text" in answer_value:
                text_val = answer_value["text"]
                if isinstance(text_val, list):
                    for item in text_val:
                        txt = _stringify(item)
                        if txt:
                            return txt
                return _stringify(text_val)
            if "answer" in answer_value:
                return _stringify(answer_value["answer"])
        if isinstance(answer_value, list):
            for item in answer_value:
                txt = _stringify(item)
                if txt:
                    return txt
            return ""
        return _stringify(answer_value)

    answer_index = _extract_field(example, spec.answer_index_field)
    if answer_index is not None and choices:
        try:
            idx = int(answer_index)
            if 0 <= idx < len(choices):
                return choices[idx]
        except Exception:
            return ""

    return ""


def _format_choices_block(choices: List[str]) -> str:
    if not choices:
        return ""
    lines = []
    for idx, choice in enumerate(choices):
        marker = chr(ord("A") + idx) if idx < 26 else str(idx + 1)
        lines.append(f"{marker}. {choice}")
    return "\n".join(lines)


def _task_defaults(task: str, language_name: str) -> Tuple[str, str]:
    if task == "summarization":
        return (
            f"Summarize the following text in {language_name}.",
            "\n\nSummary:\n",
        )
    if task == "qa":
        return (
            f"Answer the question in {language_name} using the provided context.",
            "\n\nAnswer:\n",
        )
    if task == "sentiment":
        return (
            (
                f"Classify the sentiment of the following text in {language_name} "
                "as negative, neutral, or positive."
            ),
            "\n\nSentiment:\n",
        )
    raise ValueError(f"Unsupported task: {task}")


def format_task_example(task: str, spec: TaskLanguageSpec, example: Dict[str, Any]) -> Dict[str, str] | None:
    if task not in SUPPORTED_TASKS:
        raise ValueError(f"Unsupported task: {task}")

    language_name = spec.language_name
    default_instruction, answer_suffix = _task_defaults(task, language_name)
    instruction = spec.instruction or default_instruction

    text_input = _stringify(_extract_field(example, spec.input_field))
    context = _stringify(_extract_field(example, spec.context_field))
    question = _stringify(_extract_field(example, spec.question_field))
    raw_target = _extract_field(example, spec.target_field)
    choices = _normalize_choice_list(_extract_field(example, spec.choices_field))
    choices_block = _format_choices_block(choices)

    label_text = ""
    if task == "sentiment":
        label_value = _extract_field(example, spec.label_field)
        if label_value is None:
            return None
        mapped = spec.label_map.get(str(label_value), "")
        label_text = mapped or _stringify(label_value)

    answer_text = ""
    if task == "qa":
        answer_text = _resolve_answer_text(example, spec, choices)
        if not question:
            return None

    target_text = ""
    if task == "summarization":
        target_text = _stringify(raw_target)
        if not text_input or not target_text:
            return None
    elif task == "qa":
        target_text = answer_text
        if not target_text:
            return None
    else:
        target_text = label_text
        if not text_input or not target_text:
            return None

    values = {
        "language_name": language_name,
        "instruction": instruction,
        "input": text_input,
        "context": context,
        "question": question,
        "choices": choices_block,
        "target": target_text,
    }

    system_prefix = ""
    if spec.system_prompt:
        system_prefix = spec.system_prompt.strip() + "\n\n"

    if spec.prompt_template:
        prompt = spec.prompt_template.format(**values).strip()
    else:
        if task == "summarization":
            prompt = (
                f"{system_prefix}{instruction}\n\n"
                f"Text:\n{text_input}{answer_suffix}"
            ).strip()
        elif task == "qa":
            parts = [system_prefix + instruction]
            if context:
                parts.append(f"Context:\n{context}")
            parts.append(f"Question:\n{question}")
            if choices_block:
                parts.append(f"Options:\n{choices_block}")
            parts.append("Answer:\n")
            prompt = "\n\n".join([p.strip() for p in parts if p and p.strip()]).strip()
        else:
            prompt = (
                f"{system_prefix}{instruction}\n\n"
                f"Text:\n{text_input}{answer_suffix}"
            ).strip()

    if spec.target_template:
        target = spec.target_template.format(**values).strip()
    else:
        target = target_text.strip()

    if spec.probe_template:
        probe_text = spec.probe_template.format(**values).strip()
    else:
        probe_text = prompt

    if not prompt or not target or not probe_text:
        return None
    return {
        "prompt": prompt,
        "target": target,
        "probe_text": probe_text,
    }


def resolve_task_specs(
    manifest: Dict[str, Any],
    task: str,
    requested_languages: Iterable[str] | None = None,
) -> Tuple[Dict[str, str], Dict[str, TaskLanguageSpec]]:
    if task not in manifest["tasks"]:
        raise KeyError(f"Task '{task}' not found in manifest.")

    task_block = manifest["tasks"][task]
    task_type = task_block.get("task_type", task)
    if task_type not in SUPPORTED_TASKS:
        raise ValueError(
            f"Task '{task}' declares unsupported task_type '{task_type}'. "
            f"Supported: {sorted(SUPPORTED_TASKS)}"
        )

    all_language_names = dict(DEFAULT_LANGUAGE_NAMES)
    all_language_names.update(manifest.get("languages", {}))

    requested = set(requested_languages or [])
    out: Dict[str, TaskLanguageSpec] = {}
    for lang_code, raw in task_block.get("languages", {}).items():
        if requested and lang_code not in requested:
            continue
        if not isinstance(raw, dict):
            raise ValueError(f"Manifest entry for task={task} lang={lang_code} must be an object.")
        out[lang_code] = TaskLanguageSpec(
            builder=raw.get("builder", task_block.get("builder")),
            dataset=raw["dataset"],
            config=raw.get("config"),
            data_files=_normalize_data_files(raw.get("data_files", task_block.get("data_files"))),
            builder_kwargs=_normalize_mapping(raw.get("builder_kwargs", task_block.get("builder_kwargs"))),
            expand_mode=raw.get("expand_mode", task_block.get("expand_mode")),
            train_split=_coalesce(raw.get("train_split"), task_block.get("default_train_split", "train")),
            eval_split=_coalesce(raw.get("eval_split"), task_block.get("default_eval_split", "validation")),
            test_split=raw.get("test_split", task_block.get("default_test_split")),
            language_name=_coalesce(raw.get("language_name"), all_language_names.get(lang_code, lang_code)),
            prompt_template=raw.get("prompt_template", task_block.get("prompt_template")),
            probe_template=raw.get("probe_template", task_block.get("probe_template")),
            target_template=raw.get("target_template", task_block.get("target_template")),
            system_prompt=raw.get("system_prompt", task_block.get("system_prompt")),
            instruction=raw.get("instruction", task_block.get("instruction")),
            input_field=raw.get("input_field", task_block.get("input_field")),
            target_field=raw.get("target_field", task_block.get("target_field")),
            context_field=raw.get("context_field", task_block.get("context_field")),
            question_field=raw.get("question_field", task_block.get("question_field")),
            choices_field=raw.get("choices_field", task_block.get("choices_field")),
            answer_field=raw.get("answer_field", task_block.get("answer_field")),
            answer_index_field=raw.get("answer_index_field", task_block.get("answer_index_field")),
            label_field=raw.get("label_field", task_block.get("label_field")),
            label_map={
                str(k): str(v)
                for k, v in dict(task_block.get("label_map", {}), **raw.get("label_map", {})).items()
            },
        )

    if requested and requested - set(out):
        missing = sorted(requested - set(out))
        raise KeyError(f"Manifest is missing task={task} entries for languages: {missing}")

    if not out:
        raise ValueError(f"No language configs resolved for task '{task}'.")

    language_names = {lc: spec.language_name for lc, spec in out.items()}
    return language_names, out


def _load_split(spec: TaskLanguageSpec, split_name: str) -> Dataset:
    if spec.builder:
        kwargs: Dict[str, Any] = {"split": split_name}
        if spec.data_files is not None:
            kwargs["data_files"] = spec.data_files
        if spec.builder_kwargs is not None:
            kwargs.update(spec.builder_kwargs)
        return load_dataset(spec.builder, spec.config, **kwargs)
    return load_dataset(spec.dataset, spec.config, split=split_name)


def _expand_records(example: Dict[str, Any], expand_mode: str | None) -> List[Dict[str, Any]]:
    if not expand_mode:
        return [example]

    if expand_mode == "squad_paragraphs":
        out: List[Dict[str, Any]] = []
        for para in example.get("paragraphs", []):
            context = para.get("context", "")
            for qa in para.get("qas", []):
                row = dict(qa)
                row["context"] = context
                row["title"] = example.get("title", "")
                out.append(row)
        return out

    if expand_mode == "context_qas":
        out = []
        context = example.get("context", "")
        for qa in example.get("qas", []):
            row = dict(qa)
            row["context"] = context
            out.append(row)
        return out

    raise ValueError(f"Unsupported expand_mode: {expand_mode}")


def build_task_records(
    task: str,
    spec: TaskLanguageSpec,
    split_name: str,
    limit: int,
    shuffle: bool,
    seed: int,
) -> Tuple[List[Dict[str, str]], int]:
    ds = _load_split(spec, split_name)
    raw_count = len(ds)
    if shuffle:
        ds = ds.shuffle(seed=seed)

    records: List[Dict[str, str]] = []
    for example in ds:
        for expanded in _expand_records(example, spec.expand_mode):
            record = format_task_example(task, spec, expanded)
            if record is not None:
                records.append(record)
            if limit > 0 and len(records) >= limit:
                break
        if limit > 0 and len(records) >= limit:
            break
    return records, raw_count


def load_split_count(spec: TaskLanguageSpec, split_name: str) -> int:
    return len(_load_split(spec, split_name))


def prepare_task_language_data(
    task: str,
    lang_code: str,
    spec: TaskLanguageSpec,
    probe_limit: int,
    train_limit: int,
    eval_limit: int,
    seed: int,
    build_train_records: bool = True,
    build_eval_records: bool = True,
) -> Dict[str, Any]:
    probe_records, raw_train_count = build_task_records(
        task=task,
        spec=spec,
        split_name=spec.train_split,
        limit=probe_limit,
        shuffle=False,
        seed=seed,
    )
    train_records: List[Dict[str, str]] = []
    if build_train_records:
        train_records, _ = build_task_records(
            task=task,
            spec=spec,
            split_name=spec.train_split,
            limit=train_limit,
            shuffle=True,
            seed=seed,
        )

    eval_records: List[Dict[str, str]] = []
    if build_eval_records:
        eval_records, eval_raw_count = build_task_records(
            task=task,
            spec=spec,
            split_name=spec.eval_split,
            limit=eval_limit,
            shuffle=False,
            seed=seed,
        )
    else:
        eval_raw_count = load_split_count(spec, spec.eval_split)

    if not probe_records:
        raise RuntimeError(f"No probe examples prepared for task={task} lang={lang_code}")
    if build_train_records and not train_records:
        raise RuntimeError(f"No train examples prepared for task={task} lang={lang_code}")
    if build_eval_records and not eval_records:
        raise RuntimeError(f"No eval examples prepared for task={task} lang={lang_code}")

    return {
        "lang": lang_code,
        "language": spec.language_name,
        "task": task,
        "dataset_source": spec.source_label,
        "raw_train_count": raw_train_count,
        "raw_eval_count": eval_raw_count,
        "effective_train_count": len(train_records)
        if build_train_records
        else (min(raw_train_count, train_limit) if train_limit > 0 else raw_train_count),
        "effective_eval_count": len(eval_records)
        if build_eval_records
        else (min(eval_raw_count, eval_limit) if eval_limit > 0 else eval_raw_count),
        "probe_texts": [r["probe_text"] for r in probe_records],
        "train_records": train_records,
        "eval_records": eval_records,
    }


def prepare_task_data_bundle(
    manifest_path: str | Path,
    task: str,
    target_languages: Iterable[str] | None,
    probe_limit: int,
    train_limit: int,
    eval_limit: int,
    seed: int,
    build_train_records: bool = True,
    build_eval_records: bool = True,
) -> Dict[str, Any]:
    manifest = load_task_manifest(manifest_path)
    language_names, specs = resolve_task_specs(manifest, task, target_languages)
    by_lang: Dict[str, Dict[str, Any]] = {}
    for lang_code, spec in specs.items():
        by_lang[lang_code] = prepare_task_language_data(
            task=task,
            lang_code=lang_code,
            spec=spec,
            probe_limit=probe_limit,
            train_limit=train_limit,
            eval_limit=eval_limit,
            seed=seed,
            build_train_records=build_train_records,
            build_eval_records=build_eval_records,
        )

    return {
        "manifest_path": str(Path(manifest_path)),
        "task": task,
        "language_names": language_names,
        "by_lang": by_lang,
    }
