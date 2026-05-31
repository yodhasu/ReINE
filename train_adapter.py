#!/usr/bin/env python3
"""
REInE Depth-Asymmetric Training Script (reproducible paper-ready version)

What this version adds:
  1. Explicit config-driven objective weights under loss.layer_objectives.
  2. Backward-compatible support for legacy loss.lambda_kl / loss.lambda_anchor.
  3. Run metadata export (JSON) with:
       - wall-clock train time
       - device / GPU model
       - CUDA / PyTorch / Transformers versions
       - trainable parameter count
       - dataset size / steps / batch size
       - peak VRAM allocated / reserved
       - final layer scales
       - final losses (last seen CE / KL / anchor)
  4. Optional reproducibility controls via training.deterministic.
  5. Automatic copy of resolved config used for the run.

Output files saved in save_dir:
  - adapter_final.pt
  - run_metadata.json
  - resolved_config.yaml
"""

import argparse
import json
import math
import os
import platform
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
import yaml
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, __version__ as transformers_version

from adapter import DeepAdapterWrapper


DEFAULT_OBJECTIVE_CFG = {
    "upper_ce_weight": 1.0,
    "upper_kl_weight": 0.005,
    "middle_ce_weight": 0.35,
    "middle_kl_weight": 0.005,
    "lower_ce_weight": 0.0,
    "lower_kl_weight": 0.0,
    "lower_anchor_weight": 1.0,
    "think_end_weight": 5.0,
}


def sanitize_floats(d):
    for k, v in d.items():
        if isinstance(v, dict):
            sanitize_floats(v)
        elif isinstance(v, str):
            try:
                if "." in v or "e" in v.lower():
                    d[k] = float(v)
            except ValueError:
                pass


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in {"true", "1", "yes", "y", "on"}:
        return True
    if v in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def find_subsequence_positions(sequence: List[int], pattern: List[int]) -> List[int]:
    if not pattern or len(pattern) > len(sequence):
        return []
    out = []
    m = len(pattern)
    for i in range(len(sequence) - m + 1):
        if sequence[i:i + m] == pattern:
            out.append(i)
    return out


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


def resolve_objective_cfg(cfg: dict) -> Dict[str, float]:
    out = dict(DEFAULT_OBJECTIVE_CFG)
    layer_obj = cfg.get("loss", {}).get("layer_objectives", {})
    out.update(layer_obj)

    # Backward compatibility for older configs.
    legacy_kl = cfg.get("loss", {}).get("lambda_kl", None)
    legacy_anchor = cfg.get("loss", {}).get("lambda_anchor", None)

    if legacy_kl is not None and "layer_objectives" not in cfg.get("loss", {}):
        out["upper_kl_weight"] = float(legacy_kl)
        out["middle_kl_weight"] = float(legacy_kl)

    if legacy_anchor is not None and "layer_objectives" not in cfg.get("loss", {}):
        out["lower_anchor_weight"] = float(legacy_anchor)

    return out


class ActorJSONLDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_seq_length: int = 1024):
        self.examples = []
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.examples.append(json.loads(line))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        msgs = ex.get("messages", [])
        if not msgs or msgs[-1]["role"] != "assistant":
            return self.__getitem__((idx + 1) % len(self))

        assistant_raw = msgs[-1]["content"]
        clean = re.sub(r"<latent>.*?</latent>", "", assistant_raw, flags=re.S | re.I).strip()
        history = msgs[:-1]

        try:
            prompt = self.tokenizer.apply_chat_template(
                history, tokenize=False, add_generation_prompt=True
            )
            if re.search(r"<think>\s*$", prompt, flags=re.S | re.I):
                clean = re.sub(r"^\s*<think>\s*", "", clean, flags=re.S | re.I)
        except Exception:
            prompt = ""
            for m in history:
                prompt += f"{m['role']}: {m['content']}\n"
            prompt += "assistant:\n"

        return {"prompt": prompt, "response": clean, "messages": msgs}


def build_think_aware_labels(
    seq: List[int],
    ctx_len: int,
    tokenizer,
    include_think: bool,
) -> List[int]:
    lbl = [-100] * len(seq)
    for i in range(ctx_len, len(seq)):
        lbl[i] = seq[i]

    if include_think:
        return lbl

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
                    lbl[j] = -100
            i += ts_len
            continue

        if in_think and seq[i:i + te_len] == think_end_ids:
            in_think = False
            i += te_len
            continue

        if in_think and i >= ctx_len:
            lbl[i] = -100

        i += 1

    return lbl


def build_ce_weights(labels: List[int], tokenizer, objective_cfg: Dict) -> torch.Tensor:
    T = len(labels)
    weights = torch.ones(T, dtype=torch.float32)

    supervised_positions = [i for i, tok in enumerate(labels) if tok != -100]
    if not supervised_positions:
        return weights

    supervised_tokens = [labels[i] for i in supervised_positions]
    think_end_ids = tokenizer("</think>", add_special_tokens=False)["input_ids"]
    think_end_w = float(objective_cfg.get("think_end_weight", DEFAULT_OBJECTIVE_CFG["think_end_weight"]))

    for start in find_subsequence_positions(supervised_tokens, think_end_ids):
        for j in range(start, start + len(think_end_ids)):
            pos = supervised_positions[j]
            weights[pos] = max(weights[pos], think_end_w)

    return weights


def collate_batch(batch, tokenizer, max_seq_length, objective_cfg, include_think):
    input_ids_list, labels_list, attn_list, weight_list = [], [], [], []

    for ex in batch:
        ctx = tokenizer(ex["prompt"], add_special_tokens=False)["input_ids"]
        tgt = tokenizer(ex["response"], add_special_tokens=False)["input_ids"]
        seq = ctx + tgt + [tokenizer.eos_token_id]

        if len(seq) > max_seq_length:
            seq = seq[-max_seq_length:]
            ctx_len = max(0, len(seq) - len(tgt) - 1)
        else:
            ctx_len = len(ctx)

        lbl = build_think_aware_labels(seq, ctx_len, tokenizer, include_think)
        weights = build_ce_weights(lbl, tokenizer, objective_cfg)

        input_ids_list.append(torch.tensor(seq, dtype=torch.long))
        labels_list.append(torch.tensor(lbl, dtype=torch.long))
        attn_list.append(torch.ones(len(seq), dtype=torch.long))
        weight_list.append(weights)

    return {
        "input_ids": pad_sequence(input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id),
        "labels": pad_sequence(labels_list, batch_first=True, padding_value=-100),
        "attention_mask": pad_sequence(attn_list, batch_first=True, padding_value=0),
        "ce_weights": pad_sequence(weight_list, batch_first=True, padding_value=1.0),
    }


def print_sample(ds, tokenizer, max_seq_length, objective_cfg, include_think):
    idx = random.randint(0, len(ds) - 1)
    ex = ds[idx]
    ctx = tokenizer(ex["prompt"], add_special_tokens=False)["input_ids"]
    tgt = tokenizer(ex["response"], add_special_tokens=False)["input_ids"]
    seq = ctx + tgt + [tokenizer.eos_token_id]

    if len(seq) > max_seq_length:
        seq = seq[-max_seq_length:]
        ctx_len = max(0, len(seq) - len(tgt) - 1)
    else:
        ctx_len = len(ctx)

    lbl = build_think_aware_labels(seq, ctx_len, tokenizer, include_think)
    weights = build_ce_weights(lbl, tokenizer, objective_cfg)

    think_end_ids = tokenizer("</think>", add_special_tokens=False)["input_ids"]
    supervised_positions = [i for i, tok in enumerate(lbl) if tok != -100]
    supervised_tokens = [lbl[i] for i in supervised_positions]
    think_end_set = set()
    for start in find_subsequence_positions(supervised_tokens, think_end_ids):
        think_end_set.update(supervised_positions[start:start + len(think_end_ids)])

    think_start_ids = tokenizer("<think>", add_special_tokens=False)["input_ids"]
    ts_len, te_len = len(think_start_ids), len(think_end_ids)
    in_think_at = []
    in_think = False
    i = 0
    while i < len(seq):
        if not in_think and seq[i:i + ts_len] == think_start_ids:
            in_think = True
            in_think_at.extend([True] * ts_len)
            i += ts_len
        elif in_think and seq[i:i + te_len] == think_end_ids:
            in_think = False
            in_think_at.extend([False] * te_len)
            i += te_len
        else:
            in_think_at.append(in_think)
            i += 1

    W = 130
    print("\n" + "=" * W)
    print(f"DEBUG SAMPLE (idx={idx}, include_think={include_think})")
    print("=" * W)
    print(f"{'POS':>5}  {'TOKEN_ID':>9}  {'TEXT':>28}  {'LBL':>6}  {'W':>5}  {'THINK_ZONE':>10}  {'MASKED':>7}")
    print("-" * W)
    for pos in range(len(seq)):
        tok_id = seq[pos]
        text = repr(tokenizer.decode([tok_id]))[:28]
        supervised = "YES" if lbl[pos] != -100 else "-"
        w_val = weights[pos].item()
        in_zone = "Y" if (pos < len(in_think_at) and in_think_at[pos]) else "-"
        masked = "Y" if (pos >= ctx_len and lbl[pos] == -100) else "-"
        print(
            f"{pos:>5}  {tok_id:>9}  {text:>28}  {supervised:>6}  {w_val:>5.2f}"
            f"  {'THINK':>10}" if in_zone == "Y" else
            f"{pos:>5}  {tok_id:>9}  {text:>28}  {supervised:>6}  {w_val:>5.2f}"
            f"  {'-':>10}  {masked:>7}"
        )
    print("=" * W + "\n")

    n_supervised = sum(1 for l in lbl if l != -100)
    n_masked_think = sum(
        1 for p in range(ctx_len, len(seq))
        if lbl[p] == -100 and p < len(in_think_at) and in_think_at[p]
    )
    print(f"  Supervised tokens : {n_supervised}")
    print(f"  Think-masked tokens (excluded from CE): {n_masked_think}")
    print()


def load_jsonl_messages(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_anchor_text(cfg: dict, tokenizer) -> str:
    anchor_cfg = cfg.get("anchor", {})
    mode = anchor_cfg.get("mode", "instruction_text")

    if mode == "instruction_text":
        text_path = anchor_cfg.get("text_path")
        if not text_path:
            raise ValueError("anchor.text_path is required for mode=instruction_text")
        return Path(text_path).read_text(encoding="utf-8").strip()

    if mode == "compiled_history_from_dataset":
        dataset_path = cfg["data"]["dataset_path"]
        rows = load_jsonl_messages(dataset_path)
        max_examples = int(anchor_cfg.get("max_examples", 16))
        only_assistant = bool(anchor_cfg.get("only_assistant", False))
        history = []
        for row in rows[:max_examples]:
            for m in row.get("messages", []):
                if only_assistant and m.get("role") != "assistant":
                    continue
                history.append({"role": m["role"], "content": m["content"]})
        if not history:
            raise ValueError("Compiled history anchor is empty.")
        return tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=False)

    if mode == "chat_history_file":
        path = anchor_cfg.get("text_path")
        if not path:
            raise ValueError("anchor.text_path is required for mode=chat_history_file")
        return Path(path).read_text(encoding="utf-8").strip()

    raise ValueError(f"Unknown anchor.mode: {mode}")


def get_anchor_vectors(
    model: DeepAdapterWrapper,
    tokenizer,
    device,
    anchor_text: str,
    lower_layers: List[int],
) -> Dict[int, torch.Tensor]:
    model.eval()
    model.set_adapters_enabled(False)
    model.set_capture_enabled(True)

    inputs = tokenizer(
        anchor_text, return_tensors="pt", truncation=True,
        max_length=tokenizer.model_max_length
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        _ = model(**inputs, return_dict=True)
        anchor_vecs = {}
        for idx in lower_layers:
            pooled = model.get_pooled_hidden(idx, inputs.get("attention_mask"))
            anchor_vecs[idx] = pooled.detach().mean(dim=0).float()

    model.set_adapters_enabled(True)
    model.train()
    return anchor_vecs


def cosine_anchor_loss(model, lower_layers, attention_mask, anchor_vecs):
    losses = []
    for idx in lower_layers:
        pooled = model.get_pooled_hidden(idx, attention_mask).float()
        target = anchor_vecs[idx].to(pooled.device).unsqueeze(0).expand_as(pooled)
        sim = F.cosine_similarity(pooled, target, dim=-1)
        losses.append((1.0 - sim).mean())
    return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=attention_mask.device)


def mse_anchor_loss(model, lower_layers, attention_mask, anchor_vecs):
    losses = []
    for idx in lower_layers:
        pooled = model.get_pooled_hidden(idx, attention_mask).float()
        target = anchor_vecs[idx].to(pooled.device).unsqueeze(0).expand_as(pooled)
        losses.append(F.mse_loss(pooled, target))
    return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=attention_mask.device)


def weighted_ce_loss(logits, labels, weights):
    per_token = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view_as(labels)

    valid = (labels != -100).float()
    eff_weights = weights * valid
    denom = eff_weights.sum().clamp(min=1.0)
    return (per_token * eff_weights).sum() / denom


def collect_group_params(model: DeepAdapterWrapper, group_layers: List[int]):
    params = []
    seen = set()
    for idx in group_layers:
        k = str(idx)
        if k in model.adapters:
            for p in model.adapters[k].parameters():
                if p.requires_grad and id(p) not in seen:
                    params.append(p)
                    seen.add(id(p))
        if k in model.layer_scales:
            p = model.layer_scales[k]
            if p.requires_grad and id(p) not in seen:
                params.append(p)
                seen.add(id(p))
    return params


def assign_group_grads(loss, params, retain_graph):
    if not params:
        return
    grads = torch.autograd.grad(loss, params, retain_graph=retain_graph, allow_unused=True)
    for p, g in zip(params, grads):
        if g is None:
            continue
        if p.grad is None:
            p.grad = g.detach().clone()
        else:
            p.grad.add_(g.detach())


def collect_system_metadata(device: torch.device) -> dict:
    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
        "transformers_version": transformers_version,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device": str(device),
    }
    if torch.cuda.is_available():
        idx = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        meta["gpu"] = {
            "name": torch.cuda.get_device_name(idx),
            "total_memory_bytes": int(props.total_memory),
            "multi_processor_count": int(props.multi_processor_count),
            "major": int(props.major),
            "minor": int(props.minor),
        }
    return meta


def _save_checkpoint(model, path, cfg, lower_layers, middle_layers,
                     upper_layers, target_layers, objective_cfg, half=False):
    trainable_keys = {name for name, param in model.named_parameters() if param.requires_grad}
    state = {k: v.half() if half else v for k, v in model.state_dict().items() if k in trainable_keys}
    torch.save(
        {
            "state": state,
            "cfg": cfg,
            "lower_layers": lower_layers,
            "middle_layers": middle_layers,
            "upper_layers": upper_layers,
            "target_layers": target_layers,
            "objective_cfg": objective_cfg,
        },
        path,
    )


def train(args):
    cfg = yaml.safe_load(Path(args.config).read_text())
    sanitize_floats(cfg)

    objective_cfg = resolve_objective_cfg(cfg)
    include_think = bool(cfg.get("training", {}).get("include_think", args.include_think))
    seed = int(cfg.get("training", {}).get("seed", args.seed))
    deterministic = bool(cfg.get("training", {}).get("deterministic", False))
    set_seed(seed, deterministic=deterministic)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_meta = collect_system_metadata(device)

    print(f"[REInE Depth-Asymmetric Training] device={device} include_think={include_think} seed={seed} deterministic={deterministic}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["name"],
        dtype=torch.float16,
    ).to(device)

    lower_layers = sorted(cfg["adapter"].get("lower_layers", []))
    middle_layers = sorted(cfg["adapter"].get("middle_layers", []))
    upper_layers = sorted(cfg["adapter"].get("upper_layers", []))
    target_layers = sorted(set(lower_layers + middle_layers + upper_layers))
    if not target_layers:
        raise ValueError("Provide at least one layer in adapter.lower_layers / middle_layers / upper_layers")

    raw_scale_init = cfg["adapter"].get("layer_scale_init", {})
    layer_scale_init = {int(k): float(v) for k, v in raw_scale_init.items()}

    model = DeepAdapterWrapper(
        base_model=base,
        tokenizer=tokenizer,
        rank=cfg["adapter"].get("rank", 32),
        alpha=cfg["adapter"].get("alpha", 32.0),
        dropout=cfg["adapter"].get("dropout", 0.05),
        target_layers=target_layers,
        layer_scale_init=layer_scale_init,
        capture_layers=lower_layers,
    ).to(device)

    for p in model.base.parameters():
        p.requires_grad = False

    if cfg["adapter"].get("freeze_layer_scales", False):
        model.freeze_layer_scales()

    upper_params = collect_group_params(model, upper_layers)
    middle_params = collect_group_params(model, middle_layers)
    lower_params = collect_group_params(model, lower_layers)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_param_count = sum(p.numel() for p in trainable_params)
    total_param_count = sum(p.numel() for p in model.parameters())

    print(f"Lower layers:  {lower_layers}")
    print(f"Middle layers: {middle_layers}")
    print(f"Upper layers:  {upper_layers}")
    print(f"Trainable params: {trainable_param_count:,}")
    print(f"Objective cfg: {objective_cfg}")

    print("Initial layer scales:")
    for idx in target_layers:
        print(f"  layer {idx}: {model.layer_scales[str(idx)].item():.4f}")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=cfg["optim"]["lr_adapter"],
        weight_decay=cfg["optim"].get("weight_decay", 0.0),
    )

    max_seq_length = cfg["data"].get("max_seq_length", 1024)
    ds = ActorJSONLDataset(cfg["data"]["dataset_path"], tokenizer, max_seq_length=max_seq_length)

    if args.debug:
        print_sample(ds, tokenizer, max_seq_length, objective_cfg, include_think)

    loader = DataLoader(
        ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        collate_fn=lambda b: collate_batch(b, tokenizer, max_seq_length, objective_cfg, include_think),
    )

    anchor_vecs: Dict[int, torch.Tensor] = {}
    anchor_loss_type = cfg["loss"].get("anchor_type", "cosine")

    if lower_layers and float(objective_cfg.get("lower_anchor_weight", 1.0)) > 0:
        anchor_text = build_anchor_text(cfg, tokenizer)
        if args.dump_anchor:
            Path(args.dump_anchor).write_text(anchor_text, encoding="utf-8")
            print(f"Saved anchor text to: {args.dump_anchor}")
        anchor_vecs = get_anchor_vectors(model, tokenizer, device, anchor_text, lower_layers)
        print("Built lower-layer anchor vectors from base model pass.")

    epochs = int(cfg["training"]["epochs"])
    total_steps = epochs * len(loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_steps)
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    resolved_cfg = dict(cfg)
    resolved_cfg.setdefault("training", {})
    resolved_cfg["training"]["seed"] = seed
    resolved_cfg["training"]["deterministic"] = deterministic
    resolved_cfg["training"]["include_think"] = include_think
    resolved_cfg.setdefault("loss", {})
    resolved_cfg["loss"]["layer_objectives"] = objective_cfg
    (save_dir / "resolved_config.yaml").write_text(yaml.safe_dump(resolved_cfg, sort_keys=False), encoding="utf-8")

    model.train()
    step = 0
    start_time = time.perf_counter()
    last_metrics = {"ce": None, "kl": None, "anchor": None, "upper": None, "middle": None, "lower": None}

    for epoch in range(epochs):
        for batch in loader:
            step += 1
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                model.set_adapters_enabled(False)
                base_out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    return_dict=True,
                )
                base_logits = base_out.logits[:, :-1].contiguous()
                model.set_adapters_enabled(True)

            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                return_dict=True,
            )
            shift_logits = out.logits[:, :-1].contiguous()
            shift_labels = batch["labels"][:, 1:].contiguous()
            shift_ce_weights = batch["ce_weights"][:, 1:].contiguous()

            ce_loss = weighted_ce_loss(shift_logits, shift_labels, shift_ce_weights)

            kl = F.kl_div(
                F.log_softmax(shift_logits, dim=-1),
                F.softmax(base_logits.detach(), dim=-1),
                reduction="batchmean",
            )

            if lower_layers and float(objective_cfg.get("lower_anchor_weight", 1.0)) > 0:
                if anchor_loss_type == "cosine":
                    anchor_loss = cosine_anchor_loss(model, lower_layers, batch["attention_mask"], anchor_vecs)
                elif anchor_loss_type == "mse":
                    anchor_loss = mse_anchor_loss(model, lower_layers, batch["attention_mask"], anchor_vecs)
                else:
                    raise ValueError(f"Unknown anchor_type: {anchor_loss_type}")
            else:
                anchor_loss = torch.tensor(0.0, device=device)

            upper_loss = (
                float(objective_cfg["upper_ce_weight"]) * ce_loss +
                float(objective_cfg["upper_kl_weight"]) * kl
            )
            middle_loss = (
                float(objective_cfg["middle_ce_weight"]) * ce_loss +
                float(objective_cfg["middle_kl_weight"]) * kl
            )
            lower_loss = (
                float(objective_cfg["lower_ce_weight"]) * ce_loss +
                float(objective_cfg["lower_kl_weight"]) * kl +
                float(objective_cfg["lower_anchor_weight"]) * anchor_loss
            )

            assign_group_grads(upper_loss, upper_params, retain_graph=True)
            assign_group_grads(middle_loss, middle_params, retain_graph=True)
            assign_group_grads(lower_loss, lower_params, retain_graph=False)

            torch.nn.utils.clip_grad_norm_(trainable_params, cfg["training"].get("max_grad_norm", 1.0))
            optimizer.step()
            scheduler.step()

            last_metrics = {
                "ce": float(ce_loss.item()),
                "kl": float(kl.item()),
                "anchor": float(anchor_loss.item()),
                "upper": float(upper_loss.item()),
                "middle": float(middle_loss.item()),
                "lower": float(lower_loss.item()),
            }

            if step % 10 == 0:
                scale_str = " | ".join(
                    [f"L{idx}:{model.layer_scales[str(idx)].item():.3f}" for idx in target_layers]
                )
                print(
                    f"[REInE-ASYM] epoch={epoch+1} step={step} "
                    f"ce={ce_loss.item():.4f} kl={kl.item():.4f} anchor={anchor_loss.item():.4f} "
                    f"| upper={upper_loss.item():.4f} middle={middle_loss.item():.4f} lower={lower_loss.item():.4f} "
                    f"| {scale_str}"
                )

            if step % int(cfg["training"].get("save_every_steps", 150)) == 0:
                _save_checkpoint(
                    model,
                    save_dir / f"adapter_ckpt_{step}.pt",
                    resolved_cfg,
                    lower_layers,
                    middle_layers,
                    upper_layers,
                    target_layers,
                    objective_cfg,
                    half=False,
                )

    wall_time_sec = time.perf_counter() - start_time

    _save_checkpoint(
        model,
        save_dir / "adapter_final.pt",
        resolved_cfg,
        lower_layers,
        middle_layers,
        upper_layers,
        target_layers,
        objective_cfg,
        half=True,
    )

    final_layer_scales = {str(idx): float(model.layer_scales[str(idx)].item()) for idx in target_layers}

    print("\nFinal layer scales:")
    for idx in target_layers:
        print(f"  layer {idx}: {model.layer_scales[str(idx)].item():.4f}")
    print("REInE depth-asymmetric training complete.")

    if torch.cuda.is_available():
        peak_alloc = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
    else:
        peak_alloc = 0
        peak_reserved = 0

    metadata = {
        "system": run_meta,
        "run": {
            "wall_time_seconds": wall_time_sec,
            "wall_time_minutes": wall_time_sec / 60.0,
            "save_dir": str(save_dir.resolve()),
            "dataset_path": str(Path(cfg["data"]["dataset_path"]).resolve()),
            "dataset_examples": len(ds),
            "epochs": epochs,
            "steps_total": total_steps,
            "steps_completed": step,
            "batch_size": int(cfg["training"]["batch_size"]),
            "max_seq_length": int(max_seq_length),
            "seed": seed,
            "deterministic": deterministic,
            "include_think": include_think,
        },
        "model": {
            "name": cfg["model"]["name"],
            "total_params": total_param_count,
            "trainable_params": trainable_param_count,
            "trainable_fraction": trainable_param_count / max(total_param_count, 1),
            "lower_layers": lower_layers,
            "middle_layers": middle_layers,
            "upper_layers": upper_layers,
            "final_layer_scales": final_layer_scales,
        },
        "objective": {
            "anchor_type": anchor_loss_type,
            "layer_objectives": objective_cfg,
            "final_losses": last_metrics,
        },
        "memory": {
            "peak_vram_allocated_bytes": peak_alloc,
            "peak_vram_reserved_bytes": peak_reserved,
            "peak_vram_allocated_gb": peak_alloc / (1024 ** 3),
            "peak_vram_reserved_gb": peak_reserved / (1024 ** 3),
        },
    }

    (save_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to config YAML")
    ap.add_argument("--save_dir", required=True, help="Directory to save checkpoints and metadata")
    ap.add_argument("--debug", type=str2bool, nargs="?", const=True, default=False)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dump_anchor", default="", help="Optional path to save built anchor text")
    ap.add_argument(
        "--include_think",
        type=str2bool, nargs="?", const=True, default=False,
        help=(
            "Whether to supervise think-span tokens with CE. "
            "Config key training.include_think overrides this flag."
        ),
    )
    args = ap.parse_args()
    train(args)
