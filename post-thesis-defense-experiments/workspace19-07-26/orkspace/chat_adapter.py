#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

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

    # Remove paired <think>...</think>
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I).strip()

    # If model leaked text before a closing </think>, keep only what comes after it
    if "</think>" in text:
        text = text.split("</think>", 1)[-1].strip()

    return text.strip()


def chat_adapter(checkpoint_path: str, config_path: str):
    # =========================
    # Load config
    # =========================
    cfg = yaml.safe_load(Path(config_path).read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading base model: {cfg['model']['name']}")

    # =========================
    # Tokenizer & Base Model
    # =========================
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["name"],
        torch_dtype=torch.float16,
    ).to(device)

    # =========================
    # Load checkpoint
    # =========================
    print(f"Loading adapter checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    num_layers = len(base.model.layers)

    if "target_layers" in ckpt:
        target_layers = ckpt["target_layers"]
    elif "target_layers" in cfg.get("adapter", {}):
        target_layers = cfg["adapter"]["target_layers"]
    else:
        top_n = cfg["adapter"]["top_n"]
        target_layers = list(range(num_layers - top_n, num_layers))

    raw_scale_init = cfg.get("adapter", {}).get("layer_scale_init", {})
    layer_scale_init = {int(k): float(v) for k, v in raw_scale_init.items()}

    # =========================
    # Adapter Wrapper
    # =========================
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

    # Keep adapter params and scales in fp32
    for ad in wrapper.adapters.values():
        ad.float()
    for s in wrapper.layer_scales.values():
        s.data = s.data.float()

    print(f"Using target layers: {target_layers}")
    print("Loaded layer scales:")
    for idx in target_layers:
        print(f"  layer {idx}: {wrapper.layer_scales[str(idx)].item():.4f}")

    print("\n--- ADAPTER-ONLY CHAT ---")
    print("Residual adapters ON | Frozen host model")
    print("Type 'exit' to quit.\n")

    history = []

    # Safer defaults for evaluation-style chat
    max_new_tokens = cfg.get("chat", {}).get("max_new_tokens", 512)
    do_sample = cfg.get("chat", {}).get("do_sample", False)
    temperature = cfg.get("chat", {}).get("temperature", 0.0)
    top_p = cfg.get("chat", {}).get("top_p", 1.0)

    while True:
        try:
            user = input("User: ").strip()
            if user.lower() in ("exit", "quit"):
                break
            if not user:
                continue

            history.append({"role": "user", "content": user})

            prompt = tokenizer.apply_chat_template(
                history,
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = tokenizer(prompt, return_tensors="pt").to(device)

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
                gen_kwargs.update({"do_sample": False})

            with torch.no_grad():
                # Using wrapper.base is okay because hooks are registered on base blocks.
                output_ids = wrapper.base.generate(**inputs, **gen_kwargs)

            generated_ids = output_ids[0][inputs.input_ids.shape[1]:]

            raw_reply = tokenizer.decode(
                generated_ids,
                skip_special_tokens=False,
            )

            clean_reply = tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
            )
            clean_reply = strip_thoughts(clean_reply)

            print(f"=====================\nDEBUG - Raw Generation\n{raw_reply}\n=====================")
            print(f"REINE: {clean_reply}\n")

            # Important: store only cleaned assistant reply
            history.append({"role": "assistant", "content": clean_reply})

        except KeyboardInterrupt:
            break

    print("\n[Chat ended]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to adapter_final.pt")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    chat_adapter(args.ckpt, args.config)