#!/usr/bin/env python3
"""
ReInE research LoRA training runner for Unsloth/Qwen3.

This is a standard LoRA script, not QLoRA:
  - load_in_4bit is forced to False by default
  - base model is loaded in fp16/bf16/full precision dtype, not 4-bit

Main outputs in save_dir:
  - adapter_final/                     LoRA adapter + tokenizer
  - resolved_config.yaml               Final config after auto-hyperparam resolution
  - run_metadata.json                  System/model/data/training/eval metadata
  - training_log.jsonl                 Per-log-step losses and LR
  - answers/base_before_training.jsonl Base model answers for prompt TXT
  - answers/lora_after_training.jsonl  Adapter answers for prompt TXT
  - answers/comparison.md              Human-readable before/after comparison
  - prompt_bank_copy.txt               Copy of evaluation prompts, if provided

Expected training data:
  JSONL where each row is either:
    1) {"messages": [{"role":"user", "content":"..."}, {"role":"assistant", "content":"..."}]}
    2) {"prompt": "...", "response": "..."}
    3) {"instruction": "...", "output": "..."}

Prompt TXT evaluation format:
  - one prompt per non-empty line, OR
  - multi-line prompts separated by blank lines, OR
  - prompts separated by a line containing only ---
"""

import argparse
import hashlib
import json
import math
import os
import platform
import random
import re
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

# Compatibility patch for some older torch/peft combinations.
if not hasattr(nn.Module, "set_submodule"):
    def _set_submodule(self, target: str, module: nn.Module):
        if not target:
            raise ValueError("target must be a non-empty string")
        atoms = target.split(".")
        parent = self
        if len(atoms) > 1:
            parent = self.get_submodule(".".join(atoms[:-1]))
        setattr(parent, atoms[-1], module)
    nn.Module.set_submodule = _set_submodule

from transformers import __version__ as transformers_version
from peft import __version__ as peft_version
from unsloth import FastLanguageModel

DEFAULT_MODEL_NAME = "unsloth/Qwen3-4B-Thinking-2507"
DEFAULT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

DEFAULT_CONFIG: Dict[str, Any] = {
    "model": {
        "name": DEFAULT_MODEL_NAME,
        "dtype": "auto",              # auto | bfloat16 | float16 | float32
        "trust_remote_code": True,
    },
    "data": {
        "dataset_path": None,
        "prompt_txt": None,
        "max_seq_length": 1024,
    },
    "lora": {
        "r": "auto",
        "alpha": "auto",
        "dropout": "auto",
        "target_modules": DEFAULT_TARGET_MODULES,
        "bias": "none",
        "use_rslora": False,
    },
    "training": {
        "seed": 42,
        "deterministic": False,
        "epochs": "auto",
        "batch_size": "auto",
        "gradient_accumulation_steps": "auto",
        "target_effective_batch_size": "auto",
        "gradient_checkpointing_mode": "unsloth",
        "include_think": False,
        "max_grad_norm": 1.0,
        "log_every_steps": 5,
        "save_every_steps": 0,
    },
    "optim": {
        "lr": "auto",
        "weight_decay": 0.0,
        "scheduler": "cosine",
        "warmup_ratio": 0.03,
    },
    "loss": {
        "think_end_weight": 5.0,
    },
    "generation": {
        "enabled": True,
        "max_new_tokens": 256,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "do_sample": True,
        "repetition_penalty": 1.05,
    },
    "research": {
        "project": "ReInE",
        "method": "standard_lora_baseline",
        "notes": "Full-precision LoRA baseline for comparison against ReInE residual intervention.",
    },
}


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def sanitize_floats(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: sanitize_floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_floats(v) for v in obj]
    if isinstance(obj, str):
        try:
            if "." in obj or "e" in obj.lower():
                return float(obj)
        except ValueError:
            return obj
    return obj


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = str(v).lower()
    if v in {"true", "1", "yes", "y", "on"}:
        return True
    if v in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def set_seed(seed: int, deterministic: bool = False):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass


def sha256_file(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_info(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return {"path": str(p), "exists": False}
    return {
        "path": str(p.resolve()),
        "exists": True,
        "bytes": p.stat().st_size,
        "sha256": sha256_file(str(p)),
        "modified_utc": datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat(),
    }


def find_subsequence_positions(sequence: List[int], pattern: List[int]) -> List[int]:
    if not pattern or len(pattern) > len(sequence):
        return []
    out = []
    m = len(pattern)
    for i in range(len(sequence) - m + 1):
        if sequence[i:i + m] == pattern:
            out.append(i)
    return out


class ActorJSONLDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_seq_length: int = 1024):
        self.path = path
        self.examples: List[Dict[str, Any]] = []
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                    normalized = self.normalize_example(raw)
                    if normalized is not None:
                        self.examples.append(normalized)
                except Exception as e:
                    raise ValueError(f"Bad JSONL row at {path}:{line_no}: {e}") from e

        if not self.examples:
            raise ValueError(f"No usable training examples found in {path}")

    def normalize_example(self, ex: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if "messages" in ex and isinstance(ex["messages"], list):
            msgs = ex["messages"]
            if not msgs or msgs[-1].get("role") != "assistant":
                return None
            return {"kind": "messages", "messages": msgs}

        prompt = ex.get("prompt") or ex.get("instruction") or ex.get("input") or ex.get("question")
        response = ex.get("response") or ex.get("output") or ex.get("answer") or ex.get("completion")
        if prompt is not None and response is not None:
            return {
                "kind": "prompt_response",
                "prompt": str(prompt),
                "response": str(response),
                "messages": [
                    {"role": "user", "content": str(prompt)},
                    {"role": "assistant", "content": str(response)},
                ],
            }
        return None

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        msgs = ex.get("messages", [])
        assistant_raw = msgs[-1]["content"]
        clean = re.sub(r"<latent>.*?</latent>", "", assistant_raw, flags=re.S | re.I).strip()
        history = msgs[:-1]

        try:
            prompt = self.tokenizer.apply_chat_template(
                history, tokenize=False, add_generation_prompt=True
            )
            # Qwen Thinking templates can end generation prompt with <think>.
            # If the target also begins with <think>, avoid duplicating it.
            if re.search(r"<think>\s*$", prompt, flags=re.S | re.I):
                clean = re.sub(r"^\s*<think>\s*", "", clean, flags=re.S | re.I)
        except Exception:
            prompt = ""
            for m in history:
                prompt += f"{m.get('role', 'user')}: {m.get('content', '')}\n"
            prompt += "assistant:\n"

        return {"prompt": prompt, "response": clean, "messages": msgs}


def read_prompt_txt(path: Optional[str]) -> List[str]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Prompt TXT not found: {path}")
    text = p.read_text(encoding="utf-8")
    text = text.replace("\r\n", "\n")

    # Delimiter mode: prompt blocks separated by a line containing only ---.
    if re.search(r"(?m)^\s*---\s*$", text):
        blocks = re.split(r"(?m)^\s*---\s*$", text)
        return [b.strip() for b in blocks if b.strip()]

    # Blank-line block mode.
    if "\n\n" in text:
        blocks = re.split(r"\n\s*\n", text)
        prompts = [b.strip() for b in blocks if b.strip() and not b.strip().startswith("#")]
        if prompts:
            return prompts

    # One prompt per line mode.
    prompts = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        prompts.append(line)
    return prompts


def build_think_aware_labels(seq: List[int], ctx_len: int, tokenizer, include_think: bool) -> List[int]:
    labels = [-100] * len(seq)
    for i in range(ctx_len, len(seq)):
        labels[i] = seq[i]

    if include_think:
        return labels

    think_start_ids = tokenizer("<think>", add_special_tokens=False)["input_ids"]
    think_end_ids = tokenizer("</think>", add_special_tokens=False)["input_ids"]
    ts_len = len(think_start_ids)
    te_len = len(think_end_ids)

    in_think = False
    i = 0
    while i < len(seq):
        if not in_think and seq[i:i + ts_len] == think_start_ids:
            in_think = True
            for j in range(i, min(i + ts_len, len(seq))):
                if j >= ctx_len:
                    labels[j] = -100
            i += ts_len
            continue
        if in_think and seq[i:i + te_len] == think_end_ids:
            in_think = False
            i += te_len
            continue
        if in_think and i >= ctx_len:
            labels[i] = -100
        i += 1

    return labels


def build_ce_weights(labels: List[int], tokenizer, think_end_weight: float) -> torch.Tensor:
    T = len(labels)
    weights = torch.ones(T, dtype=torch.float32)
    supervised_positions = [i for i, tok in enumerate(labels) if tok != -100]
    if not supervised_positions:
        return weights

    supervised_tokens = [labels[i] for i in supervised_positions]
    think_end_ids = tokenizer("</think>", add_special_tokens=False)["input_ids"]
    for start in find_subsequence_positions(supervised_tokens, think_end_ids):
        for j in range(start, start + len(think_end_ids)):
            if j < len(supervised_positions):
                pos = supervised_positions[j]
                weights[pos] = max(weights[pos], float(think_end_weight))
    return weights


def collate_batch(batch, tokenizer, max_seq_length, include_think, think_end_weight):
    input_ids_list, labels_list, attn_list, weight_list = [], [], [], []

    for ex in batch:
        ctx = tokenizer(ex["prompt"], add_special_tokens=False)["input_ids"]
        tgt = tokenizer(ex["response"], add_special_tokens=False)["input_ids"]
        seq = ctx + tgt + [tokenizer.eos_token_id]

        if len(seq) > max_seq_length:
            original_len = len(seq)
            seq = seq[-max_seq_length:]
            removed = original_len - len(seq)
            ctx_len = max(0, len(ctx) - removed)
        else:
            ctx_len = len(ctx)

        labels = build_think_aware_labels(seq, ctx_len, tokenizer, include_think)
        weights = build_ce_weights(labels, tokenizer, think_end_weight)

        input_ids_list.append(torch.tensor(seq, dtype=torch.long))
        labels_list.append(torch.tensor(labels, dtype=torch.long))
        attn_list.append(torch.ones(len(seq), dtype=torch.long))
        weight_list.append(weights)

    return {
        "input_ids": pad_sequence(input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id),
        "labels": pad_sequence(labels_list, batch_first=True, padding_value=-100),
        "attention_mask": pad_sequence(attn_list, batch_first=True, padding_value=0),
        "ce_weights": pad_sequence(weight_list, batch_first=True, padding_value=1.0),
    }


def weighted_ce_loss(logits, labels, weights):
    per_token = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).view_as(labels)

    valid = (labels != -100).float()
    eff_weights = weights * valid
    denom = eff_weights.sum().clamp(min=1.0)
    return (per_token * eff_weights).sum() / denom


def collect_system_metadata(device: torch.device) -> dict:
    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
        "transformers_version": transformers_version,
        "peft_version": peft_version,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device": str(device),
        "cwd": os.getcwd(),
        "argv": sys.argv,
    }
    if torch.cuda.is_available():
        idx = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        meta["gpu"] = {
            "name": torch.cuda.get_device_name(idx),
            "total_memory_bytes": int(props.total_memory),
            "total_memory_gb": int(props.total_memory) / (1024 ** 3),
            "multi_processor_count": int(props.multi_processor_count),
            "major": int(props.major),
            "minor": int(props.minor),
            "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        }
    return meta


def count_trainable_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def count_params_by_trainability(model) -> Dict[str, Any]:
    trainable = 0
    frozen = 0
    trainable_tensors = []
    for name, p in model.named_parameters():
        n = p.numel()
        if p.requires_grad:
            trainable += n
            trainable_tensors.append({"name": name, "shape": list(p.shape), "numel": n})
        else:
            frozen += n
    return {
        "trainable_params": trainable,
        "frozen_params": frozen,
        "total_params": trainable + frozen,
        "trainable_fraction": trainable / max(trainable + frozen, 1),
        "trainable_tensor_count": len(trainable_tensors),
        "trainable_tensors_preview": trainable_tensors[:50],
    }


def resolve_dtype(dtype_name: str):
    dtype_name = str(dtype_name or "auto").lower()
    if dtype_name == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return "bfloat16", torch.bfloat16
        return "float16", torch.float16
    table = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "none": None,
    }
    return dtype_name, table.get(dtype_name, torch.float16)


def infer_auto_hparams(cfg: Dict[str, Any], dataset_size: int, system_meta: Dict[str, Any]) -> Dict[str, Any]:
    cfg = deepcopy(cfg)
    gpu_mem_gb = None
    if system_meta.get("gpu"):
        gpu_mem_gb = float(system_meta["gpu"].get("total_memory_gb", 0.0))

    max_seq = int(cfg["data"].get("max_seq_length", 1024))

    # Conservative micro-batch rule for full-precision LoRA on a 4B model.
    if cfg["training"].get("batch_size") == "auto":
        if not torch.cuda.is_available():
            batch_size = 1
        elif gpu_mem_gb is not None and gpu_mem_gb >= 70 and max_seq <= 1024:
            batch_size = 4
        elif gpu_mem_gb is not None and gpu_mem_gb >= 36:
            batch_size = 2
        else:
            batch_size = 1
        cfg["training"]["batch_size"] = batch_size

    if cfg["training"].get("target_effective_batch_size") == "auto":
        if dataset_size <= 200:
            target_eff = 8
        elif dataset_size <= 800:
            target_eff = 16
        else:
            target_eff = 32
        cfg["training"]["target_effective_batch_size"] = target_eff

    if cfg["training"].get("gradient_accumulation_steps") == "auto":
        bs = int(cfg["training"]["batch_size"])
        target_eff = int(cfg["training"]["target_effective_batch_size"])
        cfg["training"]["gradient_accumulation_steps"] = max(1, math.ceil(target_eff / max(bs, 1)))

    if cfg["training"].get("epochs") == "auto":
        if dataset_size <= 200:
            epochs = 5
        elif dataset_size <= 800:
            epochs = 3
        else:
            epochs = 2
        cfg["training"]["epochs"] = epochs

    if cfg["optim"].get("lr") == "auto":
        if dataset_size <= 200:
            lr = 2e-4
        elif dataset_size <= 800:
            lr = 1.5e-4
        else:
            lr = 1e-4
        cfg["optim"]["lr"] = lr

    if cfg["lora"].get("r") == "auto":
        cfg["lora"]["r"] = 16
    if cfg["lora"].get("alpha") == "auto":
        cfg["lora"]["alpha"] = int(cfg["lora"]["r"])
    if cfg["lora"].get("dropout") == "auto":
        cfg["lora"]["dropout"] = 0.1 if dataset_size <= 1000 else 0.05

    return cfg


def build_scheduler(optimizer, scheduler_name: str, total_steps: int, warmup_ratio: float):
    scheduler_name = str(scheduler_name or "cosine").lower()
    total_steps = max(1, int(total_steps))
    warmup_steps = max(0, int(total_steps * float(warmup_ratio)))

    if scheduler_name == "constant":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)

    if scheduler_name == "linear":
        def lr_lambda(step: int):
            if warmup_steps > 0 and step < warmup_steps:
                return float(step + 1) / float(max(1, warmup_steps))
            progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 1.0 - progress)
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Default: warmup + cosine decay.
    def cosine_lambda(step: int):
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, cosine_lambda)


def maybe_for_inference(model):
    try:
        FastLanguageModel.for_inference(model)
    except Exception:
        pass
    model.eval()


def maybe_for_training(model):
    try:
        FastLanguageModel.for_training(model)
    except Exception:
        pass
    model.train()


def render_prompt(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return f"user: {prompt}\nassistant:\n"


@torch.no_grad()
def generate_answers(model, tokenizer, prompts: List[str], cfg: Dict[str, Any], out_path: Path, label: str) -> List[Dict[str, Any]]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gen_cfg = cfg.get("generation", {})
    if not prompts or not bool(gen_cfg.get("enabled", True)):
        return []

    maybe_for_inference(model)
    rows: List[Dict[str, Any]] = []
    device = model.device

    for i, prompt in enumerate(prompts, start=1):
        rendered = render_prompt(tokenizer, prompt)
        inputs = tokenizer(rendered, return_tensors="pt", add_special_tokens=False).to(device)
        started = time.perf_counter()
        output_ids = model.generate(
            **inputs,
            max_new_tokens=int(gen_cfg.get("max_new_tokens", 256)),
            do_sample=bool(gen_cfg.get("do_sample", True)),
            temperature=float(gen_cfg.get("temperature", 0.7)),
            top_p=float(gen_cfg.get("top_p", 0.9)),
            top_k=int(gen_cfg.get("top_k", 40)),
            repetition_penalty=float(gen_cfg.get("repetition_penalty", 1.05)),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        elapsed = time.perf_counter() - started
        new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
        answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        rows.append({
            "index": i,
            "phase": label,
            "prompt": prompt,
            "answer": answer,
            "new_tokens": int(new_tokens.numel()),
            "generation_seconds": elapsed,
            "generation_config": gen_cfg,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        })

    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows


def write_comparison_md(before: List[Dict[str, Any]], after: List[Dict[str, Any]], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    by_idx_before = {r["index"]: r for r in before}
    by_idx_after = {r["index"]: r for r in after}
    indices = sorted(set(by_idx_before) | set(by_idx_after))

    lines = ["# LoRA Before/After Prompt Evaluation", ""]
    for idx in indices:
        b = by_idx_before.get(idx, {})
        a = by_idx_after.get(idx, {})
        prompt = a.get("prompt") or b.get("prompt") or ""
        lines += [
            f"## Prompt {idx}",
            "",
            "**Prompt**",
            "",
            prompt,
            "",
            "**Base before training**",
            "",
            b.get("answer", ""),
            "",
            "**LoRA after training**",
            "",
            a.get("answer", ""),
            "",
            "---",
            "",
        ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def save_adapter_and_metadata(model, tokenizer, save_dir: Path, resolved_cfg: dict, metadata: dict):
    adapter_dir = save_dir / "adapter_final"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    (save_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved_cfg, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    (save_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def summarize_dataset(ds: ActorJSONLDataset) -> Dict[str, Any]:
    prompt_lens = []
    response_lens = []
    for ex in ds.examples:
        msgs = ex.get("messages", [])
        history = msgs[:-1]
        prompt_text = "\n".join(f"{m.get('role')}: {m.get('content')}" for m in history)
        response_text = msgs[-1].get("content", "") if msgs else ""
        prompt_lens.append(len(prompt_text))
        response_lens.append(len(response_text))

    def stats(xs: List[int]) -> Dict[str, float]:
        if not xs:
            return {"min": 0, "max": 0, "mean": 0}
        return {"min": min(xs), "max": max(xs), "mean": sum(xs) / len(xs)}

    return {
        "examples": len(ds),
        "prompt_char_stats": stats(prompt_lens),
        "response_char_stats": stats(response_lens),
        "row_kinds": {k: sum(1 for e in ds.examples if e.get("kind") == k) for k in sorted(set(e.get("kind") for e in ds.examples))},
    }


def train(args):
    user_cfg: Dict[str, Any] = {}
    if args.config:
        user_cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    cfg = sanitize_floats(deep_update(DEFAULT_CONFIG, user_cfg))

    # CLI overrides beat config.
    if args.dataset_path:
        cfg["data"]["dataset_path"] = args.dataset_path
    if args.prompt_txt:
        cfg["data"]["prompt_txt"] = args.prompt_txt
    if args.model_name:
        cfg["model"]["name"] = args.model_name
    if args.max_seq_length:
        cfg["data"]["max_seq_length"] = args.max_seq_length
    if args.seed is not None:
        cfg["training"]["seed"] = args.seed
    if args.include_think is not None:
        cfg["training"]["include_think"] = args.include_think

    if not cfg["data"].get("dataset_path"):
        raise ValueError("dataset_path is required, either in YAML config or via --dataset_path")

    seed = int(cfg["training"].get("seed", 42))
    deterministic = bool(cfg["training"].get("deterministic", False))
    include_think = bool(cfg["training"].get("include_think", False))
    set_seed(seed, deterministic=deterministic)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    answers_dir = save_dir / "answers"
    answers_dir.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    system_meta = collect_system_metadata(device)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    dtype_label, dtype = resolve_dtype(cfg["model"].get("dtype", "auto"))
    cfg["model"]["resolved_dtype"] = dtype_label
    cfg["model"]["name"] = cfg["model"].get("name") or DEFAULT_MODEL_NAME

    # Force standard LoRA behavior.
    cfg["quantization"] = {
        "enabled": False,
        "load_in_4bit": False,
        "note": "Standard LoRA baseline. Base model is not loaded in 4-bit.",
    }

    print(f"[ReInE LoRA] model={cfg['model']['name']} dtype={dtype_label} load_in_4bit=False device={device}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model"]["name"],
        max_seq_length=int(cfg["data"].get("max_seq_length", 1024)),
        dtype=dtype,
        load_in_4bit=False,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    ds = ActorJSONLDataset(cfg["data"]["dataset_path"], tokenizer, max_seq_length=int(cfg["data"].get("max_seq_length", 1024)))
    dataset_meta = summarize_dataset(ds)
    cfg = infer_auto_hparams(cfg, len(ds), system_meta)

    prompts = read_prompt_txt(cfg["data"].get("prompt_txt"))
    if cfg["data"].get("prompt_txt"):
        (save_dir / "prompt_bank_copy.txt").write_text(Path(cfg["data"]["prompt_txt"]).read_text(encoding="utf-8"), encoding="utf-8")

    print(f"[Data] examples={len(ds)} eval_prompts={len(prompts)}")
    print(
        "[Auto HP] "
        f"epochs={cfg['training']['epochs']} bs={cfg['training']['batch_size']} "
        f"grad_accum={cfg['training']['gradient_accumulation_steps']} "
        f"lr={cfg['optim']['lr']} r={cfg['lora']['r']} alpha={cfg['lora']['alpha']} dropout={cfg['lora']['dropout']}"
    )

    # Baseline answers before adapter training.
    before_rows: List[Dict[str, Any]] = []
    if prompts and bool(cfg.get("generation", {}).get("enabled", True)):
        before_rows = generate_answers(
            model, tokenizer, prompts, cfg,
            answers_dir / "base_before_training.jsonl",
            label="base_before_training",
        )

    # Attach standard LoRA adapter after base evaluation.
    lora_cfg = cfg.get("lora", {})
    model = FastLanguageModel.get_peft_model(
        model,
        r=int(lora_cfg.get("r", 16)),
        target_modules=lora_cfg.get("target_modules", DEFAULT_TARGET_MODULES),
        lora_alpha=int(lora_cfg.get("alpha", 16)),
        lora_dropout=float(lora_cfg.get("dropout", 0.1)),
        bias=str(lora_cfg.get("bias", "none")),
        use_gradient_checkpointing=str(cfg.get("training", {}).get("gradient_checkpointing_mode", "unsloth")),
        random_state=seed,
        use_rslora=bool(lora_cfg.get("use_rslora", False)),
        loftq_config=None,
    )

    param_meta = count_params_by_trainability(model)
    print(f"[Params] trainable={param_meta['trainable_params']:,} total={param_meta['total_params']:,} fraction={param_meta['trainable_fraction']:.8f}")

    loader = DataLoader(
        ds,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=True,
        collate_fn=lambda b: collate_batch(
            b,
            tokenizer,
            max_seq_length=int(cfg["data"].get("max_seq_length", 1024)),
            include_think=include_think,
            think_end_weight=float(cfg.get("loss", {}).get("think_end_weight", 5.0)),
        ),
    )

    training_cfg = cfg.get("training", {})
    optim_cfg = cfg.get("optim", {})
    lr = float(optim_cfg.get("lr", 2e-4))
    weight_decay = float(optim_cfg.get("weight_decay", 0.0))
    epochs = int(training_cfg.get("epochs", 3))
    grad_accum = int(training_cfg.get("gradient_accumulation_steps", 1))
    max_grad_norm = float(training_cfg.get("max_grad_norm", 1.0))
    log_every = int(training_cfg.get("log_every_steps", 5))
    save_every = int(training_cfg.get("save_every_steps", 0))

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=lr,
        weight_decay=weight_decay,
    )

    batches_total = epochs * len(loader)
    optimizer_steps_est = max(1, math.ceil(batches_total / max(grad_accum, 1)))
    scheduler = build_scheduler(
        optimizer,
        scheduler_name=str(optim_cfg.get("scheduler", "cosine")),
        total_steps=optimizer_steps_est,
        warmup_ratio=float(optim_cfg.get("warmup_ratio", 0.03)),
    )

    resolved_cfg = deepcopy(cfg)
    resolved_cfg["training"]["seed"] = seed
    resolved_cfg["training"]["deterministic"] = deterministic
    resolved_cfg["training"]["include_think"] = include_think

    training_log_path = save_dir / "training_log.jsonl"
    training_records: List[Dict[str, Any]] = []
    epoch_summaries: List[Dict[str, Any]] = []

    maybe_for_training(model)
    step = 0
    opt_step = 0
    start_time = time.perf_counter()
    last_metrics = {"ce": None, "lr": None}
    optimizer.zero_grad(set_to_none=True)

    with training_log_path.open("w", encoding="utf-8") as log_f:
        for epoch in range(epochs):
            epoch_losses: List[float] = []
            epoch_start = time.perf_counter()
            for batch in loader:
                step += 1
                batch = {k: v.to(model.device) for k, v in batch.items()}

                out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    return_dict=True,
                )
                shift_logits = out.logits[:, :-1].contiguous()
                shift_labels = batch["labels"][:, 1:].contiguous()
                shift_ce_weights = batch["ce_weights"][:, 1:].contiguous()

                ce_loss = weighted_ce_loss(shift_logits, shift_labels, shift_ce_weights)
                loss = ce_loss / grad_accum
                loss.backward()
                epoch_losses.append(float(ce_loss.item()))

                if step % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        max_grad_norm,
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    opt_step += 1

                    current_lr = optimizer.param_groups[0]["lr"]
                    last_metrics = {"ce": float(ce_loss.item()), "lr": float(current_lr)}

                    if opt_step % log_every == 0 or opt_step == 1:
                        rec = {
                            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                            "epoch": epoch + 1,
                            "step": step,
                            "optimizer_step": opt_step,
                            "ce": float(ce_loss.item()),
                            "lr": float(current_lr),
                        }
                        print(
                            f"[LORA] epoch={epoch+1}/{epochs} step={step} opt_step={opt_step} "
                            f"ce={ce_loss.item():.4f} lr={current_lr:.6g}"
                        )
                        log_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        log_f.flush()
                        training_records.append(rec)

                    if save_every > 0 and opt_step % save_every == 0:
                        ckpt_dir = save_dir / f"adapter_ckpt_{opt_step}"
                        ckpt_dir.mkdir(parents=True, exist_ok=True)
                        model.save_pretrained(ckpt_dir)
                        tokenizer.save_pretrained(ckpt_dir)

            epoch_summaries.append({
                "epoch": epoch + 1,
                "mean_ce": sum(epoch_losses) / max(len(epoch_losses), 1),
                "min_ce": min(epoch_losses) if epoch_losses else None,
                "max_ce": max(epoch_losses) if epoch_losses else None,
                "seconds": time.perf_counter() - epoch_start,
            })

        if step % grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_grad_norm,
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            opt_step += 1
            current_lr = optimizer.param_groups[0]["lr"]
            last_metrics = {"ce": float(ce_loss.item()), "lr": float(current_lr)}

    wall_time_sec = time.perf_counter() - start_time

    if torch.cuda.is_available():
        peak_alloc = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
    else:
        peak_alloc = 0
        peak_reserved = 0

    # Adapter answers after training.
    after_rows: List[Dict[str, Any]] = []
    if prompts and bool(cfg.get("generation", {}).get("enabled", True)):
        after_rows = generate_answers(
            model, tokenizer, prompts, cfg,
            answers_dir / "lora_after_training.jsonl",
            label="lora_after_training",
        )
        write_comparison_md(before_rows, after_rows, answers_dir / "comparison.md")

    metadata = {
        "system": system_meta,
        "research": cfg.get("research", {}),
        "run": {
            "script": Path(__file__).name,
            "wall_time_seconds": wall_time_sec,
            "wall_time_minutes": wall_time_sec / 60.0,
            "save_dir": str(save_dir.resolve()),
            "seed": seed,
            "deterministic": deterministic,
        },
        "files": {
            "dataset": file_info(cfg["data"].get("dataset_path")),
            "prompt_txt": file_info(cfg["data"].get("prompt_txt")),
            "config": file_info(args.config),
        },
        "dataset": dataset_meta,
        "prompt_evaluation": {
            "prompt_count": len(prompts),
            "before_path": str((answers_dir / "base_before_training.jsonl").resolve()) if before_rows else None,
            "after_path": str((answers_dir / "lora_after_training.jsonl").resolve()) if after_rows else None,
            "comparison_path": str((answers_dir / "comparison.md").resolve()) if after_rows else None,
        },
        "model": {
            "name": cfg["model"]["name"],
            "dtype": dtype_label,
            "quantization": cfg["quantization"],
            **param_meta,
            "lora": {
                "r": int(lora_cfg.get("r", 16)),
                "alpha": int(lora_cfg.get("alpha", 16)),
                "dropout": float(lora_cfg.get("dropout", 0.1)),
                "target_modules": lora_cfg.get("target_modules", DEFAULT_TARGET_MODULES),
                "bias": str(lora_cfg.get("bias", "none")),
                "use_rslora": bool(lora_cfg.get("use_rslora", False)),
            },
        },
        "training": {
            "epochs": epochs,
            "batches_per_epoch": len(loader),
            "batches_total": batches_total,
            "steps_completed": step,
            "optimizer_steps_completed": opt_step,
            "batch_size": int(training_cfg["batch_size"]),
            "gradient_accumulation_steps": grad_accum,
            "effective_batch_size": int(training_cfg["batch_size"]) * grad_accum,
            "max_seq_length": int(cfg["data"].get("max_seq_length", 1024)),
            "include_think": include_think,
            "max_grad_norm": max_grad_norm,
            "optimizer": "AdamW",
            "lr_initial": lr,
            "weight_decay": weight_decay,
            "scheduler": str(optim_cfg.get("scheduler", "cosine")),
            "warmup_ratio": float(optim_cfg.get("warmup_ratio", 0.03)),
            "epoch_summaries": epoch_summaries,
            "final_losses": last_metrics,
        },
        "memory": {
            "peak_vram_allocated_bytes": peak_alloc,
            "peak_vram_reserved_bytes": peak_reserved,
            "peak_vram_allocated_gb": peak_alloc / (1024 ** 3),
            "peak_vram_reserved_gb": peak_reserved / (1024 ** 3),
        },
    }

    save_adapter_and_metadata(model, tokenizer, save_dir, resolved_cfg, metadata)
    print(f"[Done] Standard LoRA training complete. Saved to: {save_dir.resolve()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="Optional YAML config. CLI args override it.")
    ap.add_argument("--dataset_path", default=None, help="Training JSONL path.")
    ap.add_argument("--prompt_txt", default=None, help="TXT file of evaluation prompts to answer before and after training.")
    ap.add_argument("--save_dir", required=False, default="runs/reine_lora_run", help="Directory to save adapter, answers, metadata.")
    ap.add_argument("--model_name", default=None, help=f"Default: {DEFAULT_MODEL_NAME}")
    ap.add_argument("--max_seq_length", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--include_think", type=str2bool, nargs="?", const=True, default=None)
    ap.add_argument("--print_default_config", action="store_true", help="Print default YAML config and exit.")
    args = ap.parse_args()

    if args.print_default_config:
        print(yaml.safe_dump(DEFAULT_CONFIG, sort_keys=False, allow_unicode=True))
        return

    train(args)


if __name__ == "__main__":
    main()
