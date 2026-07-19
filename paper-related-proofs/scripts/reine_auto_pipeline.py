#!/usr/bin/env python3
"""
REInE Auto Pipeline

What it does:
1. Builds variant configs for lower-focus, middle-focus, upper-focus, full, and default.
2. Generates hyperparameter trial configs (base trial + random search trials).
3. Calls the existing train_adapter.py for training.
4. Calls the existing ablation.py for zero-shot and carry-history evaluation.
5. Scores outputs with simple rule-based identity metrics for ranking.
6. Writes leaderboard, summary txt, and best configs automatically.

Designed to keep existing REInE scripts untouched.
"""

import argparse
import copy
import csv
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml
from transformers import AutoConfig


# -----------------------------
# Utilities
# -----------------------------

def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def save_yaml(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def contains_any(text: str, terms: Sequence[str]) -> bool:
    t = normalize_text(text)
    for term in terms:
        term_n = normalize_text(term)
        if term_n and term_n in t:
            return True
    return False


def loguniform_sample(rng: random.Random, low: float, high: float) -> float:
    return math.exp(rng.uniform(math.log(low), math.log(high)))


def get_num_layers(model_name: str) -> int:
    cfg = AutoConfig.from_pretrained(model_name)
    for key in ("num_hidden_layers", "n_layer", "num_layers"):
        if hasattr(cfg, key):
            return int(getattr(cfg, key))
    raise ValueError(f"Could not infer number of layers from model config: {model_name}")


def split_thirds(num_layers: int) -> Tuple[List[int], List[int], List[int]]:
    # Balanced 3-way contiguous split with remainder distributed from the front.
    sizes = [num_layers // 3] * 3
    remainder = num_layers % 3
    for i in range(remainder):
        sizes[i] += 1

    lower = list(range(0, sizes[0]))
    middle = list(range(sizes[0], sizes[0] + sizes[1]))
    upper = list(range(sizes[0] + sizes[1], num_layers))
    return lower, middle, upper


def longest_full_streak(scores: List[int]) -> int:
    best = 0
    cur = 0
    for s in scores:
        if s == 2:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def average_or_zero(vals: Sequence[float]) -> float:
    return float(sum(vals) / len(vals)) if vals else 0.0


def read_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# -----------------------------
# Scoring
# -----------------------------

@dataclass
class TargetSpec:
    identity_terms: List[str]
    name_terms: List[str]
    creator_terms: List[str]
    forbidden_terms: List[str]
    neutral_terms: List[str]


def infer_prompt_type(prompt: str) -> str:
    p = normalize_text(prompt)
    if "full name" in p or ("what is your name" in p and "who are you" not in p):
        return "name"
    if any(x in p for x in ["who made you", "who created you", "tell me about your creator", "creator?"]):
        return "creator"
    return "identity"


def score_answer(prompt: str, answer: str, spec: TargetSpec) -> int:
    ptype = infer_prompt_type(prompt)
    ans = normalize_text(answer)

    has_identity = contains_any(ans, spec.identity_terms)
    has_name = contains_any(ans, spec.name_terms)
    has_creator = contains_any(ans, spec.creator_terms)
    has_forbidden = contains_any(ans, spec.forbidden_terms)
    has_neutral = contains_any(ans, spec.neutral_terms)

    if ptype == "name":
        if has_name and not has_forbidden and not has_neutral:
            return 2
        if has_name:
            return 1
        return 0

    if ptype == "creator":
        if has_creator and not has_forbidden and not has_neutral:
            return 2
        if has_creator:
            return 1
        return 0

    # identity
    good = has_identity or has_name
    if good and not has_forbidden and not has_neutral:
        return 2
    if good:
        return 1
    return 0


def summarize_scores(records: List[dict], total_expected: Optional[int] = None) -> Dict[str, Any]:
    scores = [int(r["score"]) for r in records]
    total = total_expected if total_expected is not None else len(scores)
    max_points = 2 * max(total, 1)

    normalized_accuracy = sum(scores) / max_points
    full_correct = sum(1 for s in scores if s == 2)
    streak = longest_full_streak(scores)

    phase_ranges = {
        "establishment": (1, 3),
        "role_shift": (4, 7),
        "impersonation": (8, 11),
        "suppression_recovery": (12, 15),
    }

    per_phase = {}
    for name, (start, end) in phase_ranges.items():
        phase_rows = [r for r in records if start <= int(r["prompt_id"]) <= end]
        if phase_rows:
            per_phase[name] = {
                "num_prompts": len(phase_rows),
                "score_sum": sum(int(r["score"]) for r in phase_rows),
                "normalized_accuracy": sum(int(r["score"]) for r in phase_rows) / (2 * len(phase_rows)),
                "full_correct": sum(1 for r in phase_rows if int(r["score"]) == 2),
            }

    later_phases = []
    for phase_name in ("impersonation", "suppression_recovery"):
        if phase_name in per_phase:
            later_phases.append(per_phase[phase_name]["normalized_accuracy"])

    return {
        "num_prompts": len(scores),
        "score_sum": sum(scores),
        "normalized_accuracy": normalized_accuracy,
        "full_correct": full_correct,
        "longest_full_streak": streak,
        "longest_full_streak_norm": streak / max(total, 1),
        "late_phase_recovery": average_or_zero(later_phases),
        "per_phase": per_phase,
    }


def score_ablation_file(clean_jsonl: Path, spec: TargetSpec) -> Dict[str, Any]:
    rows = read_jsonl(clean_jsonl)
    scored = []
    for row in rows:
        score = score_answer(row["user_prompt"], row["clean_answer"], spec)
        enriched = dict(row)
        enriched["score"] = score
        enriched["prompt_type"] = infer_prompt_type(row["user_prompt"])
        scored.append(enriched)
    summary = summarize_scores(scored)
    return {"records": scored, "summary": summary}


# -----------------------------
# Trial configuration
# -----------------------------

@dataclass
class SearchSpace:
    lr_min: float = 1e-4
    lr_max: float = 8e-4
    upper_kl_min: float = 1e-4
    upper_kl_max: float = 1e-2
    middle_kl_min: float = 1e-4
    middle_kl_max: float = 1e-2
    lower_anchor_min: float = 0.25
    lower_anchor_max: float = 2.0
    epochs_choices: Tuple[int, ...] = (3, 5, 8)
    rank_choices: Tuple[int, ...] = (8, 16, 32)
    alpha_choices: Tuple[float, ...] = (8.0, 16.0, 32.0)
    dropout_choices: Tuple[float, ...] = (0.05, 0.10)


@dataclass
class TrialParams:
    lr_adapter: float
    epochs: int
    rank: int
    alpha: float
    dropout: float
    upper_kl_weight: float
    middle_kl_weight: float
    lower_anchor_weight: float


@dataclass
class TrialResult:
    variant: str
    trial_id: str
    config_path: str
    train_dir: str
    zs_accuracy: float
    history_accuracy: float
    history_streak_norm: float
    history_late_phase_recovery: float
    objective_score: float
    trainable_params: Optional[int]
    wall_time_minutes: Optional[float]
    success: bool
    error: Optional[str]


class Pipeline:
    def __init__(self, args):
        self.args = args
        self.base_config_path = Path(args.base_config).resolve()
        self.train_script = Path(args.train_script).resolve()
        self.ablation_script = Path(args.ablation_script).resolve()
        self.prompts_path = Path(args.prompts).resolve()
        self.outdir = Path(args.outdir).resolve()
        self.python_bin = args.python_bin or sys.executable
        self.rng = random.Random(args.seed)
        self.base_cfg = load_yaml(self.base_config_path)
        self.model_name = self.base_cfg["model"]["name"]
        self.num_layers = get_num_layers(self.model_name)
        self.lower_third, self.middle_third, self.upper_third = split_thirds(self.num_layers)

        self.outdir.mkdir(parents=True, exist_ok=True)
        self.configs_dir = self.outdir / "configs"
        self.trials_dir = self.outdir / "trials"
        self.summaries_dir = self.outdir / "summaries"
        for p in (self.configs_dir, self.trials_dir, self.summaries_dir):
            p.mkdir(parents=True, exist_ok=True)

        self.spec = TargetSpec(
            identity_terms=[x.strip() for x in args.target_identity.split("||") if x.strip()],
            name_terms=[x.strip() for x in args.target_name.split("||") if x.strip()],
            creator_terms=[x.strip() for x in args.target_creator.split("||") if x.strip()],
            forbidden_terms=[x.strip() for x in args.forbidden_terms.split("||") if x.strip()],
            neutral_terms=[x.strip() for x in args.neutral_terms.split("||") if x.strip()],
        )
        self.search = SearchSpace()

    def sample_params(self, use_base: bool = False) -> TrialParams:
        if use_base:
            lo = self.base_cfg.get("loss", {}).get("layer_objectives", {})
            return TrialParams(
                lr_adapter=float(self.base_cfg["optim"]["lr_adapter"]),
                epochs=int(self.base_cfg["training"]["epochs"]),
                rank=int(self.base_cfg["adapter"].get("rank", 16)),
                alpha=float(self.base_cfg["adapter"].get("alpha", 16.0)),
                dropout=float(self.base_cfg["adapter"].get("dropout", 0.1)),
                upper_kl_weight=float(lo.get("upper_kl_weight", 0.001)),
                middle_kl_weight=float(lo.get("middle_kl_weight", 0.001)),
                lower_anchor_weight=float(lo.get("lower_anchor_weight", 1.0)),
            )

        return TrialParams(
            lr_adapter=loguniform_sample(self.rng, self.search.lr_min, self.search.lr_max),
            epochs=self.rng.choice(self.search.epochs_choices),
            rank=self.rng.choice(self.search.rank_choices),
            alpha=float(self.rng.choice(self.search.alpha_choices)),
            dropout=float(self.rng.choice(self.search.dropout_choices)),
            upper_kl_weight=loguniform_sample(self.rng, self.search.upper_kl_min, self.search.upper_kl_max),
            middle_kl_weight=loguniform_sample(self.rng, self.search.middle_kl_min, self.search.middle_kl_max),
            lower_anchor_weight=self.rng.uniform(self.search.lower_anchor_min, self.search.lower_anchor_max),
        )

    def apply_layer_variant(self, cfg: dict, variant: str) -> dict:
        cfg = copy.deepcopy(cfg)
        adapter = cfg.setdefault("adapter", {})
        loss = cfg.setdefault("loss", {})
        lo = loss.setdefault("layer_objectives", {})

        if variant == "default":
            return cfg

        if variant == "lower_focus":
            adapter["lower_layers"] = self.lower_third
            adapter["middle_layers"] = []
            adapter["upper_layers"] = []
            adapter["layer_scale_init"] = {str(i): 1.0 for i in self.lower_third}
            return cfg

        if variant == "middle_focus":
            adapter["lower_layers"] = []
            adapter["middle_layers"] = self.middle_third
            adapter["upper_layers"] = []
            adapter["layer_scale_init"] = {str(i): 1.0 for i in self.middle_third}
            lo["lower_anchor_weight"] = 0.0
            return cfg

        if variant == "upper_focus":
            adapter["lower_layers"] = []
            adapter["middle_layers"] = []
            adapter["upper_layers"] = self.upper_third
            adapter["layer_scale_init"] = {str(i): 1.0 for i in self.upper_third}
            lo["lower_anchor_weight"] = 0.0
            return cfg

        if variant == "full":
            # Full depth-symmetric coverage using all layers, uniform scales,
            # and symmetric CE/KL weights with anchor disabled.
            adapter["lower_layers"] = self.lower_third
            adapter["middle_layers"] = self.middle_third
            adapter["upper_layers"] = self.upper_third
            adapter["layer_scale_init"] = {str(i): 1.0 for i in range(self.num_layers)}

            base_upper_ce = float(lo.get("upper_ce_weight", 1.0))
            base_upper_kl = float(lo.get("upper_kl_weight", 0.001))
            lo["upper_ce_weight"] = base_upper_ce
            lo["middle_ce_weight"] = base_upper_ce
            lo["lower_ce_weight"] = base_upper_ce
            lo["upper_kl_weight"] = base_upper_kl
            lo["middle_kl_weight"] = base_upper_kl
            lo["lower_kl_weight"] = base_upper_kl
            lo["lower_anchor_weight"] = 0.0
            return cfg

        raise ValueError(f"Unknown variant: {variant}")

    def apply_trial_params(self, cfg: dict, params: TrialParams, variant: str) -> dict:
        cfg = copy.deepcopy(cfg)
        cfg.setdefault("optim", {})["lr_adapter"] = float(params.lr_adapter)
        cfg.setdefault("training", {})["epochs"] = int(params.epochs)
        cfg.setdefault("adapter", {})["rank"] = int(params.rank)
        cfg["adapter"]["alpha"] = float(params.alpha)
        cfg["adapter"]["dropout"] = float(params.dropout)

        lo = cfg.setdefault("loss", {}).setdefault("layer_objectives", {})
        lo["upper_kl_weight"] = float(params.upper_kl_weight)
        lo["middle_kl_weight"] = float(params.middle_kl_weight)

        if variant != "full":
            lo["lower_anchor_weight"] = float(params.lower_anchor_weight)
        return cfg

    def call_subprocess(self, cmd: List[str], cwd: Optional[Path] = None) -> None:
        print("$", " ".join(cmd))
        subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)

    def run_training(self, config_path: Path, train_dir: Path) -> None:
        cmd = [
            self.python_bin,
            str(self.train_script),
            "--config", str(config_path),
            "--save_dir", str(train_dir),
        ]
        self.call_subprocess(cmd)

    def run_ablation(self, ckpt: Path, config_path: Path, outdir: Path, carry_history: bool, tag: str) -> Path:
        cmd = [
            self.python_bin,
            str(self.ablation_script),
            "--ckpt", str(ckpt),
            "--config", str(config_path),
            "--prompts", str(self.prompts_path),
            "--outdir", str(outdir),
            "--tag", tag,
        ]
        if carry_history:
            cmd.append("--carry_history")
        self.call_subprocess(cmd)

        matches = sorted(outdir.glob(f"{tag}_*_clean.jsonl"))
        if not matches:
            raise FileNotFoundError(f"No clean ablation JSONL found under {outdir} for tag={tag}")
        return matches[-1]

    def objective(self, zs_summary: Dict[str, Any], hist_summary: Dict[str, Any]) -> float:
        zs_acc = float(zs_summary["normalized_accuracy"])
        hist_acc = float(hist_summary["normalized_accuracy"])
        hist_streak = float(hist_summary["longest_full_streak_norm"])
        hist_recovery = float(hist_summary["late_phase_recovery"])

        # History gets more weight because your main stress story is sequential persistence.
        return (
            0.30 * zs_acc
            + 0.45 * hist_acc
            + 0.15 * hist_streak
            + 0.10 * hist_recovery
        )

    def one_trial(self, variant: str, trial_idx: int, use_base: bool) -> TrialResult:
        trial_id = f"{variant}_trial_{trial_idx:03d}_{'base' if use_base else 'rand'}"
        trial_root = self.trials_dir / variant / trial_id
        train_dir = trial_root / "train"
        eval_zs_dir = trial_root / "eval_zs"
        eval_hist_dir = trial_root / "eval_history"
        config_path = self.configs_dir / variant / f"{trial_id}.yaml"
        trial_root.mkdir(parents=True, exist_ok=True)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            params = self.sample_params(use_base=use_base)
            cfg = copy.deepcopy(self.base_cfg)
            cfg = self.apply_layer_variant(cfg, variant)
            cfg = self.apply_trial_params(cfg, params, variant)
            save_yaml(cfg, config_path)
            save_json({"variant": variant, "trial_id": trial_id, "params": asdict(params)}, trial_root / "trial_params.json")

            self.run_training(config_path, train_dir)
            ckpt = train_dir / "adapter_final.pt"
            if not ckpt.exists():
                raise FileNotFoundError(f"Missing checkpoint: {ckpt}")

            zs_clean = self.run_ablation(ckpt, config_path, eval_zs_dir, carry_history=False, tag=f"{trial_id}_zs")
            hist_clean = self.run_ablation(ckpt, config_path, eval_hist_dir, carry_history=True, tag=f"{trial_id}_hist")

            zs_scored = score_ablation_file(zs_clean, self.spec)
            hist_scored = score_ablation_file(hist_clean, self.spec)
            save_json(zs_scored, eval_zs_dir / "scored_summary.json")
            save_json(hist_scored, eval_hist_dir / "scored_summary.json")

            train_meta_path = train_dir / "run_metadata.json"
            train_meta = json.loads(train_meta_path.read_text(encoding="utf-8")) if train_meta_path.exists() else {}

            score = self.objective(zs_scored["summary"], hist_scored["summary"])
            combined = {
                "variant": variant,
                "trial_id": trial_id,
                "config_path": str(config_path),
                "train_dir": str(train_dir),
                "params": asdict(params),
                "zs": zs_scored["summary"],
                "history": hist_scored["summary"],
                "objective_score": score,
                "train_metadata": train_meta,
            }
            save_json(combined, trial_root / "trial_result.json")

            trainable_params = train_meta.get("model", {}).get("trainable_params")
            wall_time_minutes = train_meta.get("run", {}).get("wall_time_minutes")

            return TrialResult(
                variant=variant,
                trial_id=trial_id,
                config_path=str(config_path),
                train_dir=str(train_dir),
                zs_accuracy=float(zs_scored["summary"]["normalized_accuracy"]),
                history_accuracy=float(hist_scored["summary"]["normalized_accuracy"]),
                history_streak_norm=float(hist_scored["summary"]["longest_full_streak_norm"]),
                history_late_phase_recovery=float(hist_scored["summary"]["late_phase_recovery"]),
                objective_score=float(score),
                trainable_params=trainable_params,
                wall_time_minutes=wall_time_minutes,
                success=True,
                error=None,
            )
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            save_json({"variant": variant, "trial_id": trial_id, "error": err}, trial_root / "trial_error.json")
            return TrialResult(
                variant=variant,
                trial_id=trial_id,
                config_path=str(config_path),
                train_dir=str(train_dir),
                zs_accuracy=0.0,
                history_accuracy=0.0,
                history_streak_norm=0.0,
                history_late_phase_recovery=0.0,
                objective_score=-1.0,
                trainable_params=None,
                wall_time_minutes=None,
                success=False,
                error=err,
            )

    def write_variant_summary(self, variant: str, results: List[TrialResult]) -> None:
        variant_dir = self.summaries_dir / variant
        variant_dir.mkdir(parents=True, exist_ok=True)

        ordered = sorted(results, key=lambda x: x.objective_score, reverse=True)
        save_json([asdict(r) for r in ordered], variant_dir / "results.json")

        with (variant_dir / "leaderboard.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(ordered[0]).keys()) if ordered else [])
            if ordered:
                writer.writeheader()
                for row in ordered:
                    writer.writerow(asdict(row))

        if ordered and ordered[0].success:
            best = ordered[0]
            shutil.copy2(best.config_path, variant_dir / "best_config.yaml")
            summary_txt = [
                f"Variant: {variant}",
                f"Best trial: {best.trial_id}",
                f"Objective score: {best.objective_score:.6f}",
                f"ZS accuracy: {best.zs_accuracy:.6f}",
                f"History accuracy: {best.history_accuracy:.6f}",
                f"History streak norm: {best.history_streak_norm:.6f}",
                f"History late-phase recovery: {best.history_late_phase_recovery:.6f}",
                f"Trainable params: {best.trainable_params}",
                f"Wall time minutes: {best.wall_time_minutes}",
                f"Config: {best.config_path}",
                f"Train dir: {best.train_dir}",
            ]
            (variant_dir / "summary.txt").write_text("\n".join(summary_txt), encoding="utf-8")

    def write_global_summary(self, all_results: List[TrialResult]) -> None:
        ordered = sorted(all_results, key=lambda x: x.objective_score, reverse=True)
        save_json([asdict(r) for r in ordered], self.summaries_dir / "global_leaderboard.json")

        with (self.summaries_dir / "global_leaderboard.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(ordered[0]).keys()) if ordered else [])
            if ordered:
                writer.writeheader()
                for row in ordered:
                    writer.writerow(asdict(row))

        lines = [
            "REInE Auto Pipeline Summary",
            f"Base config: {self.base_config_path}",
            f"Train script: {self.train_script}",
            f"Ablation script: {self.ablation_script}",
            f"Prompts: {self.prompts_path}",
            f"Model: {self.model_name}",
            f"Total layers: {self.num_layers}",
            f"Lower third: {self.lower_third}",
            f"Middle third: {self.middle_third}",
            f"Upper third: {self.upper_third}",
            "",
        ]
        for row in ordered[:10]:
            lines.append(
                f"{row.variant:12s} | {row.trial_id:30s} | score={row.objective_score:.6f} "
                f"| zs={row.zs_accuracy:.4f} | hist={row.history_accuracy:.4f} "
                f"| streak={row.history_streak_norm:.4f} | recovery={row.history_late_phase_recovery:.4f}"
            )
        (self.summaries_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")

    def run(self) -> None:
        variants = [x.strip() for x in self.args.variants.split(",") if x.strip()]
        manifest = {
            "base_config": str(self.base_config_path),
            "train_script": str(self.train_script),
            "ablation_script": str(self.ablation_script),
            "prompts": str(self.prompts_path),
            "model": self.model_name,
            "num_layers": self.num_layers,
            "lower_third": self.lower_third,
            "middle_third": self.middle_third,
            "upper_third": self.upper_third,
            "variants": variants,
            "trials_per_variant": self.args.trials_per_variant,
            "seed": self.args.seed,
            "target_spec": asdict(self.spec),
        }
        save_json(manifest, self.outdir / "manifest.json")

        all_results: List[TrialResult] = []
        for variant in variants:
            print(f"\n{'='*80}\nRunning variant: {variant}\n{'='*80}")
            variant_results = []
            total_trials = max(1, self.args.trials_per_variant)
            for idx in range(total_trials):
                use_base = idx == 0
                result = self.one_trial(variant=variant, trial_idx=idx, use_base=use_base)
                variant_results.append(result)
                all_results.append(result)
                print(asdict(result))
            self.write_variant_summary(variant, variant_results)

        self.write_global_summary(all_results)
        print(f"\nDone. See: {self.summaries_dir}")


# -----------------------------
# CLI
# -----------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Auto-run REInE training + ablation + random hyperparam search.")
    ap.add_argument("--base-config", required=True, help="Base config YAML")
    ap.add_argument("--train-script", default="train_adapter.py", help="Path to existing training script")
    ap.add_argument("--ablation-script", default="ablation.py", help="Path to existing ablation script")
    ap.add_argument("--prompts", required=True, help="Prompt .txt file for ablation")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--trials-per-variant", type=int, default=4, help="Includes 1 base trial + N-1 random trials")
    ap.add_argument("--variants", default="lower_focus,middle_focus,upper_focus,full,default")
    ap.add_argument("--python-bin", default="", help="Python executable to use. Default: current interpreter")
    ap.add_argument("--seed", type=int, default=42)

    # Target specification for rule-based scoring.
    ap.add_argument("--target-identity", required=True, help="Pipe-separated target identity aliases using '||'")
    ap.add_argument("--target-name", required=True, help="Pipe-separated target full-name aliases using '||'")
    ap.add_argument("--target-creator", required=True, help="Pipe-separated creator aliases using '||'")
    ap.add_argument(
        "--forbidden-terms",
        default="qwen||alibaba||neutral ai||no name||nameless assistant",
        help="Pipe-separated forbidden identity terms",
    )
    ap.add_argument(
        "--neutral-terms",
        default="neutral ai||no name||i don't have a name||i do not have a name||nameless",
        help="Pipe-separated neutral/suppression terms considered wrong for identity prompts",
    )
    return ap


def main():
    parser = build_parser()
    args = parser.parse_args()
    pipeline = Pipeline(args)
    pipeline.run()


if __name__ == "__main__":
    main()
