#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from adapter import DeepAdapterWrapper


def strip_thoughts(text: str) -> str:
    """
    Remove visible reasoning sections from generated text.

    Handles cases like:
    - <think> ... </think> final answer
    - reasoning ... </think> final answer
    """
    if not text:
        return text

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I).strip()

    if "</think>" in text:
        text = text.split("</think>", 1)[-1].strip()

    return text.strip()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_json_dump(obj, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def load_prompts(prompt_path: Path) -> List[Dict]:
    """
    Prompt file formats supported:

    1. One prompt per non-empty line
    2. Multi-line prompts separated by a line containing only ---
    3. Lines starting with # are ignored as comments
    """
    raw = prompt_path.read_text(encoding="utf-8")

    if "\n---\n" in raw or raw.strip().startswith("---") or raw.strip().endswith("---"):
        chunks = re.split(r"(?m)^\s*---\s*$", raw)
        prompts = []
        for i, chunk in enumerate(chunks, start=1):
            text = chunk.strip()
            if not text:
                continue
            prompts.append({"id": i, "prompt": text})
        return prompts

    prompts = []
    idx = 1
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        prompts.append({"id": idx, "prompt": line})
        idx += 1
    return prompts


def build_generation_kwargs(cfg: dict, tokenizer) -> Dict:
    chat_cfg = cfg.get("chat", {})
    max_new_tokens = int(chat_cfg.get("max_new_tokens", 512))
    do_sample = bool(chat_cfg.get("do_sample", False))
    temperature = float(chat_cfg.get("temperature", 0.0))
    top_p = float(chat_cfg.get("top_p", 1.0))

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
    }

    if do_sample:
        gen_kwargs.update(
            {
                "do_sample": True,
                "temperature": temperature,
                "top_p": top_p,
            }
        )
    else:
        gen_kwargs["do_sample"] = False

    return gen_kwargs


def resolve_target_layers(cfg: dict, ckpt: dict, base) -> List[int]:
    num_layers = len(base.model.layers)

    if "target_layers" in ckpt:
        return ckpt["target_layers"]

    adapter_cfg = cfg.get("adapter", {})
    if "target_layers" in adapter_cfg:
        return adapter_cfg["target_layers"]

    if "top_n" in adapter_cfg:
        top_n = int(adapter_cfg["top_n"])
        return list(range(num_layers - top_n, num_layers))

    lower_layers = sorted(adapter_cfg.get("lower_layers", []))
    middle_layers = sorted(adapter_cfg.get("middle_layers", []))
    upper_layers = sorted(adapter_cfg.get("upper_layers", []))
    target_layers = sorted(set(lower_layers + middle_layers + upper_layers))

    if not target_layers:
        raise ValueError("Could not resolve target layers from checkpoint or config.")
    return target_layers


def load_model_and_wrapper(ckpt_path: Path, config_path: Path):
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading base model: {cfg['model']['name']}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["name"],
        torch_dtype=torch.float16,
    ).to(device)

    print(f"Loading adapter checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    target_layers = resolve_target_layers(cfg, ckpt, base)

    raw_scale_init = cfg.get("adapter", {}).get("layer_scale_init", {})
    layer_scale_init = {int(k): float(v) for k, v in raw_scale_init.items()}

    wrapper = DeepAdapterWrapper(
        base_model=base,
        tokenizer=tokenizer,
        rank=cfg["adapter"].get("rank", 32),
        alpha=cfg["adapter"].get("alpha", 32.0),
        dropout=cfg["adapter"].get("dropout", 0.05),
        target_layers=target_layers,
        layer_scale_init=layer_scale_init,
    )

    wrapper.load_state_dict(ckpt["state"], strict=False)
    wrapper.to(device)
    wrapper.eval()

    for ad in wrapper.adapters.values():
        ad.float()
    for s in wrapper.layer_scales.values():
        s.data = s.data.float()

    print(f"Using target layers: {target_layers}")
    print("Loaded layer scales:")
    for idx in target_layers:
        print(f"  layer {idx}: {wrapper.layer_scales[str(idx)].item():.4f}")

    return cfg, tokenizer, base, wrapper, device, ckpt, target_layers


def generate_reply(
    wrapper,
    tokenizer,
    device,
    history: List[Dict[str, str]],
    user_prompt: str,
    gen_kwargs: Dict,
) -> Tuple[str, str, str, int]:
    local_history = history + [{"role": "user", "content": user_prompt}]

    prompt_text = tokenizer.apply_chat_template(
        local_history,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = wrapper.base.generate(**inputs, **gen_kwargs)

    generated_ids = output_ids[0][inputs.input_ids.shape[1]:]
    raw_reply = tokenizer.decode(generated_ids, skip_special_tokens=False)
    decoded_reply = tokenizer.decode(generated_ids, skip_special_tokens=True)
    clean_reply = strip_thoughts(decoded_reply)

    return prompt_text, raw_reply, clean_reply, int(generated_ids.shape[0])


def main():
    ap = argparse.ArgumentParser(description="Run REInE identity ablation prompts automatically.")
    ap.add_argument("--ckpt", required=True, help="Path to adapter_final.pt")
    ap.add_argument("--config", required=True, help="Path to config.yaml")
    ap.add_argument("--prompts", required=True, help="Path to prompt .txt file")
    ap.add_argument("--outdir", required=True, help="Directory to save results")
    ap.add_argument(
        "--carry_history",
        action="store_true",
        help="If set, run prompts as one continuous conversation. Default: each prompt is independent.",
    )
    ap.add_argument(
        "--tag",
        default="identity_ablation",
        help="Optional experiment tag for metadata and filenames.",
    )
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt).resolve()
    config_path = Path(args.config).resolve()
    prompt_path = Path(args.prompts).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(prompt_path)
    if not prompts:
        raise ValueError(f"No prompts found in: {prompt_path}")

    cfg, tokenizer, base, wrapper, device, ckpt, target_layers = load_model_and_wrapper(
        ckpt_path, config_path
    )

    gen_kwargs = build_generation_kwargs(cfg, tokenizer)

    run_id = f"{args.tag}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    metadata_path = outdir / f"{run_id}_metadata.json"
    clean_path = outdir / f"{run_id}_clean.jsonl"
    raw_path = outdir / f"{run_id}_raw.jsonl"

    layer_scales = {
        str(idx): float(wrapper.layer_scales[str(idx)].item())
        for idx in target_layers
    }

    metadata = {
        "run_id": run_id,
        "tag": args.tag,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "script_name": Path(__file__).name if "__file__" in globals() else "identity_ablation.py",
        "python_version": sys.version,
        "platform": platform.platform(),
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "torch_version": torch.__version__,
        "model_name": cfg["model"]["name"],
        "checkpoint_path": str(ckpt_path),
        "checkpoint_sha256": sha256_file(ckpt_path),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "prompts_path": str(prompt_path),
        "prompts_sha256": sha256_file(prompt_path),
        "num_prompts": len(prompts),
        "carry_history": bool(args.carry_history),
        "generation_kwargs": gen_kwargs,
        "target_layers": target_layers,
        "loaded_layer_scales": layer_scales,
        "adapter_cfg": cfg.get("adapter", {}),
        "loss_cfg": cfg.get("loss", {}),
        "training_cfg": cfg.get("training", {}),
        "optim_cfg": cfg.get("optim", {}),
        "checkpoint_keys": sorted(list(ckpt.keys())),
    }
    safe_json_dump(metadata, metadata_path)

    clean_f = clean_path.open("w", encoding="utf-8")
    raw_f = raw_path.open("w", encoding="utf-8")

    shared_history: List[Dict[str, str]] = []

    print(f"\nRunning {len(prompts)} prompts...")
    print(f"Metadata: {metadata_path}")
    print(f"Clean results: {clean_path}")
    print(f"Raw results: {raw_path}\n")

    for row in prompts:
        prompt_id = row["id"]
        user_prompt = row["prompt"]

        history = shared_history if args.carry_history else []

        prompt_text, raw_reply, clean_reply, gen_len = generate_reply(
            wrapper=wrapper,
            tokenizer=tokenizer,
            device=device,
            history=history,
            user_prompt=user_prompt,
            gen_kwargs=gen_kwargs,
        )

        clean_record = {
            "run_id": run_id,
            "prompt_id": prompt_id,
            "user_prompt": user_prompt,
            "clean_answer": clean_reply,
        }

        raw_record = {
            "run_id": run_id,
            "prompt_id": prompt_id,
            "user_prompt": user_prompt,
            "prompt_text_used": prompt_text,
            "raw_generation": raw_reply,
            "clean_answer": clean_reply,
            "generated_token_count": gen_len,
        }

        clean_f.write(json.dumps(clean_record, ensure_ascii=False) + "\n")
        raw_f.write(json.dumps(raw_record, ensure_ascii=False) + "\n")

        print(f"[{prompt_id}/{len(prompts)}] done")

        if args.carry_history:
            shared_history.append({"role": "user", "content": user_prompt})
            shared_history.append({"role": "assistant", "content": clean_reply})

    clean_f.close()
    raw_f.close()

    print("\nFinished.")
    print(f"Metadata saved to: {metadata_path}")
    print(f"Clean results saved to: {clean_path}")
    print(f"Raw results saved to: {raw_path}")


if __name__ == "__main__":
    main()