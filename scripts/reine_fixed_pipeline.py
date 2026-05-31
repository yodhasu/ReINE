#!/usr/bin/env python3
"""
Fixed-variant REInE experiment pipeline.

What it does:
1. Generate fixed configs for these variants:
   - lower_third
   - middle_third
   - upper_third
   - full_symmetric
   - default
2. Call the existing training script for each config.
3. Call the existing ablation script twice for each trained checkpoint:
   - zero-shot / stateless
   - carry-history
4. Aggregate metadata and file paths for all runs.

It does NOT score outputs. Validation is done later.

Typical usage:
python reine_fixed_pipeline.py \
  --base-config config.yaml \
  --train-script train_adapter.py \
  --ablation-script ablation.py \
  --prompts stress_test.txt \
  --outdir runs/fixed_variants \
  --variant-overrides variant_overrides.yaml

Optional override file format:
variants:
  lower_third:
    adapter:
      rank: 16
      alpha: 16.0
    optim:
      lr_adapter: 5e-4
    training:
      epochs: 5
  default:
    training:
      epochs: 8

Notes:
- `default` keeps the base config exactly as-is, except for any explicit overrides.
- `lower_third`, `middle_third`, `upper_third` each target exactly one third of the host layers.
- `full_symmetric` targets all layers and, by default, uses symmetric layer scales and objective weights.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from transformers import AutoConfig


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def write_yaml(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def infer_num_layers(model_name: str) -> int:
    cfg = AutoConfig.from_pretrained(model_name)
    for attr in ("num_hidden_layers", "n_layer", "num_layers", "n_layers"):
        if hasattr(cfg, attr):
            value = getattr(cfg, attr)
            if isinstance(value, int) and value > 0:
                return value
    raise RuntimeError(f"Could not infer number of transformer layers for model: {model_name}")


def split_layers(num_layers: int) -> Tuple[List[int], List[int], List[int]]:
    layers = list(range(num_layers))
    third = num_layers // 3
    rem = num_layers % 3

    lower_n = third + (1 if rem > 0 else 0)
    middle_n = third + (1 if rem > 1 else 0)
    upper_n = num_layers - lower_n - middle_n

    lower = layers[:lower_n]
    middle = layers[lower_n: lower_n + middle_n]
    upper = layers[lower_n + middle_n:]
    return lower, middle, upper


def make_scale_init(layers: List[int], value: float) -> Dict[str, float]:
    return {str(i): float(value) for i in layers}


def default_symmetric_objectives() -> Dict[str, float]:
    return {
        "upper_ce_weight": 1.0,
        "upper_kl_weight": 0.001,
        "middle_ce_weight": 1.0,
        "middle_kl_weight": 0.001,
        "lower_ce_weight": 1.0,
        "lower_kl_weight": 0.001,
        "lower_anchor_weight": 0.0,
        "think_end_weight": 5.0,
    }


def build_variant_config(base_cfg: Dict[str, Any], variant: str, num_layers: int) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    adapter_cfg = cfg.setdefault("adapter", {})
    loss_cfg = cfg.setdefault("loss", {})
    loss_cfg.setdefault("layer_objectives", {})

    lower, middle, upper = split_layers(num_layers)
    all_layers = list(range(num_layers))

    if variant == "default":
        return cfg

    if variant == "lower_third":
        adapter_cfg["lower_layers"] = lower
        adapter_cfg["middle_layers"] = []
        adapter_cfg["upper_layers"] = []
        adapter_cfg["layer_scale_init"] = make_scale_init(lower, 1.0)
        return cfg

    if variant == "middle_third":
        adapter_cfg["lower_layers"] = []
        adapter_cfg["middle_layers"] = middle
        adapter_cfg["upper_layers"] = []
        adapter_cfg["layer_scale_init"] = make_scale_init(middle, 1.0)
        return cfg

    if variant == "upper_third":
        adapter_cfg["lower_layers"] = []
        adapter_cfg["middle_layers"] = []
        adapter_cfg["upper_layers"] = upper
        adapter_cfg["layer_scale_init"] = make_scale_init(upper, 1.0)
        return cfg

    if variant == "full_symmetric":
        adapter_cfg["lower_layers"] = lower
        adapter_cfg["middle_layers"] = middle
        adapter_cfg["upper_layers"] = upper
        adapter_cfg["layer_scale_init"] = make_scale_init(all_layers, 1.0)
        cfg["loss"]["layer_objectives"] = default_symmetric_objectives()
        return cfg

    raise ValueError(f"Unknown variant: {variant}")


def load_variant_overrides(path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    if path is None:
        return {}
    data = read_yaml(path)
    if not isinstance(data, dict):
        raise ValueError("variant override YAML must be a dict")
    return data.get("variants", {}) or {}


def run_command(cmd: List[str], cwd: Optional[Path], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as logf:
        logf.write("$ " + " ".join(cmd) + "\n\n")
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            logf.write(line)
        code = proc.wait()
        if code != 0:
            raise subprocess.CalledProcessError(code, cmd)


def latest_matching(path: Path, pattern: str) -> Optional[Path]:
    matches = sorted(path.glob(pattern), key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None


def collect_eval_artifacts(eval_dir: Path) -> Dict[str, Optional[str]]:
    meta = latest_matching(eval_dir, "*_metadata.json")
    clean = latest_matching(eval_dir, "*_clean.jsonl")
    raw = latest_matching(eval_dir, "*_raw.jsonl")
    return {
        "metadata_json": str(meta) if meta else None,
        "clean_jsonl": str(clean) if clean else None,
        "raw_jsonl": str(raw) if raw else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Fixed-variant REInE train + ablation pipeline")
    ap.add_argument("--base-config", required=True, help="Base config YAML")
    ap.add_argument("--train-script", required=True, help="Path to train_adapter.py")
    ap.add_argument("--ablation-script", required=True, help="Path to ablation.py")
    ap.add_argument("--prompts", required=True, help="Path to ablation prompt .txt")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--variant-overrides", default="", help="Optional YAML with per-variant overrides")
    ap.add_argument("--python", default=sys.executable, help="Python executable to use")
    ap.add_argument("--seed", type=int, default=42, help="Override training seed for all variants")
    ap.add_argument("--debug-train", action="store_true", help="Pass --debug true to training script")
    ap.add_argument("--include-think", default="", help="Optional override for training --include_think (true/false)")
    args = ap.parse_args()

    base_config_path = Path(args.base_config).resolve()
    train_script = Path(args.train_script).resolve()
    ablation_script = Path(args.ablation_script).resolve()
    prompts_path = Path(args.prompts).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    base_cfg = read_yaml(base_config_path)
    model_name = base_cfg["model"]["name"]
    num_layers = infer_num_layers(model_name)

    overrides = load_variant_overrides(Path(args.variant_overrides).resolve() if args.variant_overrides else None)
    variants = ["lower_third", "middle_third", "upper_third", "full_symmetric", "default"]

    manifest: Dict[str, Any] = {
        "created_at_utc": now_utc(),
        "python": args.python,
        "platform": platform.platform(),
        "base_config": str(base_config_path),
        "train_script": str(train_script),
        "ablation_script": str(ablation_script),
        "prompts": str(prompts_path),
        "model_name": model_name,
        "num_layers": num_layers,
        "variants": {},
    }

    summary_rows: List[Dict[str, Any]] = []

    for variant in variants:
        print("\n" + "=" * 88)
        print(f"[PIPELINE] Variant: {variant}")
        print("=" * 88)

        variant_cfg = build_variant_config(base_cfg, variant, num_layers)
        if args.seed is not None:
            variant_cfg.setdefault("training", {})["seed"] = int(args.seed)
        variant_cfg = deep_update(variant_cfg, overrides.get(variant, {}))

        variant_dir = outdir / variant
        config_dir = variant_dir / "config"
        train_dir = variant_dir / "train"
        eval_zs_dir = variant_dir / "eval_zs"
        eval_hist_dir = variant_dir / "eval_history"
        logs_dir = variant_dir / "logs"

        config_path = config_dir / f"{variant}.yaml"
        write_yaml(config_path, variant_cfg)

        train_cmd = [
            args.python,
            str(train_script),
            "--config", str(config_path),
            "--save_dir", str(train_dir),
            "--seed", str(variant_cfg.get("training", {}).get("seed", args.seed)),
        ]
        if args.debug_train:
            train_cmd.extend(["--debug", "true"])
        if args.include_think:
            train_cmd.extend(["--include_think", args.include_think])

        run_command(train_cmd, cwd=train_script.parent, log_path=logs_dir / "train.log")

        ckpt_path = train_dir / "adapter_final.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Expected checkpoint not found: {ckpt_path}")

        zs_cmd = [
            args.python,
            str(ablation_script),
            "--ckpt", str(ckpt_path),
            "--config", str(config_path),
            "--prompts", str(prompts_path),
            "--outdir", str(eval_zs_dir),
            "--tag", f"{variant}_zs",
        ]
        run_command(zs_cmd, cwd=ablation_script.parent, log_path=logs_dir / "ablation_zs.log")

        hist_cmd = [
            args.python,
            str(ablation_script),
            "--ckpt", str(ckpt_path),
            "--config", str(config_path),
            "--prompts", str(prompts_path),
            "--outdir", str(eval_hist_dir),
            "--carry_history",
            "--tag", f"{variant}_history",
        ]
        run_command(hist_cmd, cwd=ablation_script.parent, log_path=logs_dir / "ablation_history.log")

        run_metadata_path = train_dir / "run_metadata.json"
        resolved_config_path = train_dir / "resolved_config.yaml"

        train_meta = read_yaml(run_metadata_path) if False else None
        if run_metadata_path.exists():
            train_meta = json.loads(run_metadata_path.read_text(encoding="utf-8"))

        record = {
            "variant": variant,
            "variant_dir": str(variant_dir),
            "generated_config": str(config_path),
            "resolved_config": str(resolved_config_path) if resolved_config_path.exists() else None,
            "checkpoint": str(ckpt_path),
            "train_metadata": str(run_metadata_path) if run_metadata_path.exists() else None,
            "eval_zs": collect_eval_artifacts(eval_zs_dir),
            "eval_history": collect_eval_artifacts(eval_hist_dir),
            "train_log": str(logs_dir / "train.log"),
            "ablation_zs_log": str(logs_dir / "ablation_zs.log"),
            "ablation_history_log": str(logs_dir / "ablation_history.log"),
        }

        if train_meta:
            record["train_summary"] = {
                "dataset_examples": train_meta.get("run", {}).get("dataset_examples"),
                "epochs": train_meta.get("run", {}).get("epochs"),
                "steps_completed": train_meta.get("run", {}).get("steps_completed"),
                "trainable_params": train_meta.get("model", {}).get("trainable_params"),
                "trainable_fraction": train_meta.get("model", {}).get("trainable_fraction"),
                "wall_time_minutes": train_meta.get("run", {}).get("wall_time_minutes"),
                "peak_vram_allocated_gb": train_meta.get("memory", {}).get("peak_vram_allocated_gb"),
                "peak_vram_reserved_gb": train_meta.get("memory", {}).get("peak_vram_reserved_gb"),
            }

        manifest["variants"][variant] = record

        row = {
            "variant": variant,
            "config": str(config_path),
            "checkpoint": str(ckpt_path),
            "train_metadata": str(run_metadata_path) if run_metadata_path.exists() else "",
            "zs_clean": record["eval_zs"]["clean_jsonl"] or "",
            "zs_raw": record["eval_zs"]["raw_jsonl"] or "",
            "history_clean": record["eval_history"]["clean_jsonl"] or "",
            "history_raw": record["eval_history"]["raw_jsonl"] or "",
        }
        if train_meta:
            row.update({
                "dataset_examples": train_meta.get("run", {}).get("dataset_examples", ""),
                "epochs": train_meta.get("run", {}).get("epochs", ""),
                "trainable_params": train_meta.get("model", {}).get("trainable_params", ""),
                "wall_time_minutes": train_meta.get("run", {}).get("wall_time_minutes", ""),
                "peak_vram_allocated_gb": train_meta.get("memory", {}).get("peak_vram_allocated_gb", ""),
            })
        summary_rows.append(row)

    manifest_path = outdir / "manifest.json"
    write_json(manifest_path, manifest)

    csv_path = outdir / "experiment_index.csv"
    fieldnames = []
    for row in summary_rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    summary_txt = outdir / "summary.txt"
    lines = [
        "REInE fixed-variant pipeline summary",
        f"Created: {now_utc()}",
        f"Model: {model_name}",
        f"Num layers: {num_layers}",
        f"Base config: {base_config_path}",
        f"Train script: {train_script}",
        f"Ablation script: {ablation_script}",
        f"Prompts: {prompts_path}",
        "",
    ]
    for row in summary_rows:
        lines.extend([
            f"[{row['variant']}]",
            f"  config: {row['config']}",
            f"  checkpoint: {row['checkpoint']}",
            f"  train metadata: {row['train_metadata']}",
            f"  zs clean: {row['zs_clean']}",
            f"  zs raw: {row['zs_raw']}",
            f"  history clean: {row['history_clean']}",
            f"  history raw: {row['history_raw']}",
            "",
        ])
    summary_txt.write_text("\n".join(lines), encoding="utf-8")

    print("\nDone.")
    print(f"Manifest: {manifest_path}")
    print(f"CSV index: {csv_path}")
    print(f"Summary: {summary_txt}")


if __name__ == "__main__":
    main()
