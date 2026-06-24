# ReINE Folder and Run Grouping

This map groups the extracted `SkripsiYodha` dump using the current thesis draft `Thesis_Draft_REINE-4.pdf` as the narrative source of truth, then cross-checks it against manifests, configs, metadata, and folder contents.

## Source Code to Build the GitHub Repo Around

Use `SkripsiYodha/ReInE-main/ReInE-main` as the core repo seed, but do not treat every file in the dump as source.

Core ReINE implementation:

- `ReInE-main/ReInE-main/adapter.py`
- `ReInE-main/ReInE-main/train_adapter.py`
- `ReInE-main/ReInE-main/chat_adapter.py`
- `ReInE-main/ReInE-main/config.yaml`
- `ReInE-main/ReInE-main/config600v2.yaml`
- `ReInE-main/ReInE-main/configlow.yaml`
- `ReInE-main/ReInE-main/instruction_prompt.txt`
- `ReInE-main/ReInE-main/test_prompt.txt`
- `ReInE-main/ReInE-main/tunedataset*.jsonl`

Outer scripts that should be brought into the repo because they explain later experiments:

- `train_lora_unsloth_reine.py` - standard LoRA baseline runner, not QLoRA.
- `reine_fixed_pipeline.py` - fixed-variant pipeline for lower/middle/upper/full/default suites.
- `reine_auto_pipeline.py` - exploratory pipeline with trial generation and scoring.

Supporting evaluation code:

- `ReInE-main/ReInE-main/ablation/ablation.py` - REInE identity ablation evaluator.
- `ReInE-main/ReInE-main/qlora/*.py` - older QLoRA-related code; keep as legacy or experimental unless it still matches the thesis.

## Zip Files and Experiment Families

| Zip or folder | Group | Role | Thesis relationship | Git treatment |
| --- | --- | --- | --- | --- |
| `ReInE-main.zip` | Source archive | Original source package | Not an experiment result | Do not commit zip; commit cleaned extracted source |
| `runs.zip` | Initial fixed-variant suite | April 30 fixed variants on 157-example dataset | Table I | Split into `experiments/initial_157_fixed_variants`; omit `.pt` checkpoints |
| `runs (1).zip` | Superset bundle | Contains byte-identical `runs.zip` contents plus `600_reine` and `reine_lora_baseline` | Tables I, II, IV, V | Do not commit zip; split unique contents by family |
| `run-11-1-1.zip` | Lower-focused probe | Layers 0-10, 20, 27; no CoT | Table III, 4/30 | Keep metadata/eval outputs; omit `.pt` |
| `run-lower5.zip` | Lower-focused probe | Layers 0-4; no CoT | Table III, 18/30 | Keep metadata/eval outputs; omit `.pt` |
| `run-lower5-cot.zip` | Main best ReINE probe | Layers 0-4; CoT supervised | Abstract/Table III/Table IV/V, 30/30 | Keep metadata/eval outputs; omit `.pt` |
| `run-0.zip` | Single-layer probe | Layer 0 only; CoT supervised | Results paragraph, 28/30 | Keep metadata/eval outputs; omit `.pt` |
| `ablation_runs_*.zip` under `ReInE-main/ablation` | Early qualitative probes | March 31 intended/lower/upper prompt outputs with and without carry-history | Pre-thesis/old evidence | Archive under `experiments/legacy_2026-03-31_ablation_probes` or omit from public repo |

## Thesis-Core Experiment Groups

### 1. Initial 157-Example Layer-Composition Ablation

Canonical source:

- `runs/runs`
- also duplicated byte-for-byte at the root of `runs (1)/runs`

Generated from:

- `config.yaml`
- `train_adapter.py`
- `ablation.py`
- `test_prompt.txt`

Variants:

- `lower_third`
- `middle_third`
- `upper_third`
- `full_symmetric`
- `default`

Draft Table I scores:

- Lower Third: 24/30
- Middle Third: 3/30
- Upper Third: 12/30
- Full Symmetric: 11/30
- Default Asymmetric: 24/30

Keep:

- `manifest.json`
- `experiment_index.csv`
- `summary.txt`
- per-variant configs, metadata, logs, clean/raw eval JSONL

Do not commit normally:

- `adapter_final.pt`
- `adapter_ckpt_*.pt`

### 2. 665-Example Fixed-Variant Follow-Up

Canonical source:

- `runs (1)/runs/600_reine`

Generated from:

- `config600v2.yaml`
- `train_adapter.py`
- `ablation.py`
- `test_prompt.txt`
- dataset recorded as `tunedataset_600v2.jsonl` in resolved configs and metadata

Variants:

- `lower_third`
- `middle_third`
- `upper_third`
- `full_symmetric`
- `default`

Draft Table II scores:

- Lower Third: 28/30
- Default Asymmetric: 21/30

Important reconciliation note:

- `1-3-lower-600-variant.txt` says Lower Third scored 21/30.
- `Depth-Asymetric-(default)-600-experiment-result.txt` says Default Asymmetric scored 13/30.
- The current draft says Lower Third 28/30 and Default Asymmetric 21/30.
- Treat the draft as current, but before publishing the repo, reconcile which human-judgment notes are final versus outdated.

### 3. Lower-Focused 665-Example Probes

Canonical sources:

- `run-11-1-1/run-11-1-1`
- `run-lower5/run-lower5`
- `run-lower5-cot/run-lower5-cot`
- `run-0/run-0`

All use:

- `unsloth/Qwen3-4B-Thinking-2507`
- `tunedataset_600v2.jsonl`
- 5 epochs
- zero-shot 15-prompt identity ablation

Draft Table III and paragraph:

- `run-11-1-1`: layers 0-10, 20, 27; no CoT; 4/30.
- `run-lower5`: layers 0-4; no CoT; 18/30.
- `run-lower5-cot`: layers 0-4; CoT supervised; 30/30.
- `run-0`: layer 0 only; CoT supervised; 28/30.

This is the main follow-up family supporting the thesis claim that shallow lower-layer intervention is strongest.

### 4. Standard LoRA Baseline

Canonical source:

- `runs (1)/runs/reine_lora_baseline`

Generated from:

- `train_lora_unsloth_reine.py`
- `reine_lora_default_config.yaml`
- `tunedataset_600v2.jsonl`
- `test_prompt.txt`

Draft Tables IV and V:

- LoRA score: 22/30, accuracy 0.733.
- Trainable parameters: 33,030,144.
- Time: 5.16 min.
- Peak VRAM: 8.71 GB.

Keep:

- `run_metadata.json`
- `resolved_config.yaml`
- `training_log.jsonl`
- `prompt_bank_copy.txt`
- `answers/base_before_training.jsonl`
- `answers/lora_after_training.jsonl`
- `answers/comparison.md`
- `adapter_final/adapter_config.json`
- `adapter_final/README.md`

Do not commit normally:

- `adapter_final/adapter_model.safetensors`
- `adapter_final/tokenizer.json` unless you explicitly want a model-artifact release or Git LFS.

## Exploratory or Legacy Material

### March 31 Ablation Probe Zips

Files:

- `ablation_runs_intended.zip`
- `ablation_runs_intended_history.zip`
- `ablation_runs_lower.zip`
- `ablation_runs_lower_history.zip`
- `ablation_runs_upper.zip`
- `ablation_runs_upper_history.zip`

These contain only three files each: raw JSONL, clean JSONL, and metadata. They are early qualitative identity-ablation probes against `/workspace/intended`, `/workspace/lower`, and `/workspace/upper` checkpoints. They are useful for provenance but not part of the final thesis tables.

### Older QLoRA Folder

Folder:

- `ReInE-main/ReInE-main/qlora`

Contents:

- `train_qlora_unsloth.py`
- `ablation_qlora.py`
- `qlora_unsloth_config.yaml`

The current draft compares against a standard LoRA baseline, not QLoRA. Keep this folder only if you want to document an abandoned/legacy experiment path.

## Recommended GitHub Layout

```text
ReINE/
  README.md
  LICENSE
  requirements.txt
  src/
    reine/
      adapter.py
  scripts/
    train_adapter.py
    chat_adapter.py
    ablation.py
    train_lora_unsloth_reine.py
    reine_fixed_pipeline.py
    reine_auto_pipeline.py
  configs/
    reine_default.yaml
    reine_600v2.yaml
    reine_lower5_cot.yaml
    lora_baseline.yaml
  data/
    tunedataset.jsonl
    tunedataset_600.jsonl
    tunedataset_600v2.jsonl
  prompts/
    instruction_prompt.txt
    test_prompt.txt
  experiments/
    initial_157_fixed_variants/
    followup_665_fixed_variants/
    lower_focused_665_probes/
    lora_baseline_665/
    legacy_2026-03-31_ablation_probes/
  docs/
    paper/
    experiment_notes/
```

## Recommended `.gitignore`

```gitignore
__pycache__/
*.py[cod]

# Large model artifacts
*.pt
*.safetensors
adapter_ckpt_*.pt
adapter_final.pt
tokenizer.json

# Raw archives and downloaded metadata
*.zip
*.Identifier
*:Zone.Identifier

# Local/generated run folders
runs/
run-*/
outputs/
work/

# Optional paper build outputs
*.aux
*.bbl
*.blg
*.log
*.out
```

## Cleanup Priorities

1. Make a clean repo root from `ReInE-main/ReInE-main`.
2. Bring in the three outer scripts: `train_lora_unsloth_reine.py`, `reine_fixed_pipeline.py`, and `reine_auto_pipeline.py`.
3. Move prompts and configs into named folders so experiment configs are not confused with generated resolved configs.
4. Split run outputs into the four experiment families above.
5. Keep metadata, manifests, summaries, notes, configs, logs, and eval JSONL in Git.
6. Exclude model weights/checkpoints from Git unless using Git LFS or a release artifact.
7. Reconcile the 665-example score mismatch between the current draft and older handwritten notes before publishing.
