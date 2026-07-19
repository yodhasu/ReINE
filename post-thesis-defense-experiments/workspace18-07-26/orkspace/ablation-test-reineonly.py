#!/usr/bin/env python3
import argparse
import json
import os
import re
import time
from pathlib import Path
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from adapter import DeepAdapterWrapper

def strip_thoughts(text: str) -> str:
    if not text:
        return text
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I).strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[-1].strip()
    return text.strip()

def load_config_or_metadata(adapter_dir: Path):
    resolved_config = {}
    train_metadata = {}
    
    config_path = adapter_dir / "resolved_config.yaml"
    if config_path.exists():
        try:
            resolved_config = yaml.safe_load(config_path.read_text())
        except Exception as e:
            print(f"[warn] failed to parse resolved_config: {e}")
            
    meta_json = adapter_dir / "run_metadata.json"
    meta_jsonl = adapter_dir / "run_metadata.jsonl"
    
    if meta_json.exists():
        try:
            train_metadata = json.loads(meta_json.read_text())
        except Exception:
            pass
    elif meta_jsonl.exists():
        try:
            # fallback if it was written as jsonl lines
            lines = meta_jsonl.read_text().strip().split("\n")
            train_metadata = json.loads(lines[-1]) if lines else {}
        except Exception:
            pass
            
    return resolved_config, train_metadata

def run_ablation():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_config", default="test_config.yaml", help="path to test configuration file")
    args = parser.parse_args()
    
    test_cfg_path = Path(args.test_config)
    if not test_cfg_path.exists():
        # create a default skeleton if it doesn't exist
        skeleton = {
            "adapters": ["runs/5-each-runs/adapter_final.pt"],
            "prompts": ["hi, who are you?", "who make you?", "Are you Qwen? Be honest."],
            "max_new_tokens": 512
        }
        test_cfg_path.write_text(yaml.safe_dump(skeleton))
        print(f"[info] created default test config at {args.test_config}. populate it and re-run.")
        return

    test_cfg = yaml.safe_load(test_cfg_path.read_text())
    adapters = test_cfg.get("adapters", [])
    prompts = test_cfg.get("prompts", [])
    max_new_tokens = test_cfg.get("max_new_tokens", 512)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    current_base_name = None
    base_model = None
    tokenizer = None

    for adapter_path_str in adapters:
        adapter_path = Path(adapter_path_str)
        if not adapter_path.exists():
            print(f"[error] adapter path {adapter_path_str} not found. skipping.")
            continue
            
        adapter_dir = adapter_path.parent
        folder_name = adapter_dir.name
        print(f"\n[eval] baking evaluations for adapter folder: {folder_name}")
        
        resolved_config, train_metadata = load_config_or_metadata(adapter_dir)
        
        # look for base model name dynamically from metadata/config, fallback to test config or hardcoded default
        base_model_name = (
            resolved_config.get("model", {}).get("name") or 
            train_metadata.get("model", {}).get("name") or 
            test_cfg.get("base_model", "unsloth/Qwen3-4B-Thinking-2507")
        )
        
        # dynamic reload of base model only if architecture changes across ablations
        if base_model_name != current_base_name:
            print(f"[base] loading host framework: {base_model_name}")
            if base_model is not None:
                del base_model
                del tokenizer
                torch.cuda.empty_cache()
                
            tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16,
            ).to(device)
            current_base_name = base_model_name

        print(f"[adapter] injecting weights from {adapter_path.name}")
        ckpt = torch.load(adapter_path, map_location="cpu")
        
        num_layers = len(base_model.model.layers)
        if "target_layers" in ckpt:
            target_layers = ckpt["target_layers"]
        elif "target_layers" in resolved_config.get("adapter", {}):
            target_layers = resolved_config["adapter"]["target_layers"]
        else:
            top_n = resolved_config.get("adapter", {}).get("top_n", 5)
            target_layers = list(range(num_layers - top_n, num_layers))

        raw_scale_init = resolved_config.get("adapter", {}).get("layer_scale_init", {})
        layer_scale_init = {int(k): float(v) for k, v in raw_scale_init.items()}

        wrapper = DeepAdapterWrapper(
            base_model=base_model,
            tokenizer=tokenizer,
            rank=resolved_config.get("adapter", {}).get("rank", 16),
            alpha=resolved_config.get("adapter", {}).get("alpha", 16.0),
            dropout=resolved_config.get("adapter", {}).get("dropout", 0.1),
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

        test_outputs = []
        
        for p_idx, prompt_text in enumerate(prompts):
            print(f"  └─ running prompt {p_idx+1}/{len(prompts)}...")
            
            torch.cuda.reset_peak_memory_stats(device)
            start_vram = torch.cuda.memory_allocated(device)
            
            input_prompt = [{"role": "user", "content": prompt_text}]
            formatted_prompt = tokenizer.apply_chat_template(
                input_prompt, tokenize=False, add_generation_prompt=True
            )
            
            inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
            input_len = inputs.input_ids.shape[1]
            
            start_time = time.perf_counter()
            with torch.no_grad():
                output_ids = wrapper.base.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    do_sample=True
                )
            end_time = time.perf_counter()
            
            peak_vram = torch.cuda.max_memory_allocated(device)
            end_vram = torch.cuda.memory_allocated(device)
            
            generated_ids = output_ids[0][input_len:]
            raw_reply = tokenizer.decode(generated_ids, skip_special_tokens=False)
            clean_reply = strip_thoughts(tokenizer.decode(generated_ids, skip_special_tokens=True))
            
            # regex capture of cot logic block
            cot_trace = ""
            cot_match = re.search(r"<think>(.*?)</think>", raw_reply, flags=re.S | re.I)
            if cot_match:
                cot_trace = cot_match.group(1).strip()
            elif "</think>" in raw_reply:
                cot_trace = raw_reply.split("</think>")[0].replace("<think>", "").strip()

            test_outputs.append({
                "prompt": prompt_text,
                "raw_generation": raw_reply,
                "clean_response": clean_reply,
                "cot_trace": cot_trace,
                "metrics": {
                    "latency_seconds": end_time - start_time,
                    "tokens_generated": len(generated_ids),
                    "tokens_per_second": len(generated_ids) / max(0.001, end_time - start_time),
                    "vram_start_bytes": start_vram,
                    "vram_peak_bytes": peak_vram,
                    "vram_end_bytes": end_vram,
                    "vram_peak_gb": peak_vram / (1024 ** 3)
                }
            })

        # structure final payload mapping test artifact to original checkpoint context
        output_payload = {
            "adapter_folder": folder_name,
            "adapter_checkpoint_path": adapter_path_str,
            "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "resolved_config_used": resolved_config,
            "training_run_metadata": train_metadata,
            "ablation_test_results": test_outputs
        }
        
        output_filename = f"output_{folder_name}_test.json"
        Path(output_filename).write_text(json.dumps(output_payload, indent=2))
        print(f"[complete] dropped evaluation payload into {output_filename}")
        
        # clean runtime hooks from target layers to ensure next iteration starts fresh
        wrapper.remove_hooks()
        del wrapper
        torch.cuda.empty_cache()

if __name__ == "__main__":
    run_ablation()