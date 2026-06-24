# ReINE

ReINE (**Residual Information Network Editing**) is a frozen-host persona steering experiment for causal language models. The host model stays frozen while small trainable residual MicroAdapters are attached to selected transformer layers through forward hooks.

This branch adds a cleaned research-evidence prep package for the ReINE thesis/research dump.

## Main Code

The core implementation is kept at the repository root:

- `adapter.py` - residual MicroAdapter wrapper and hidden-state intervention logic.
- `train_adapter.py` - ReINE adapter training script.
- `chat_adapter.py` - interactive inference script using a trained adapter.
- `config*.yaml` - ReINE training configurations.
- `instruction_prompt.txt` and `test_prompt.txt` - identity/persona prompting and stress-test prompts.

Additional and legacy scripts are in `scripts/` and `result/` from the earlier repository state.

## Research Evidence Prep

The cleaned prep material is in:

- [`research_evidence_prep/`](research_evidence_prep/)

Start here:

- [`research_evidence_prep/result_notes.md`](research_evidence_prep/result_notes.md) - detailed rough lab notes for all grouped runs.
- [`research_evidence_prep/docs/experiment_notes/run_grouping.md`](research_evidence_prep/docs/experiment_notes/run_grouping.md) - folder/run grouping map.
- [`research_evidence_prep/dataset_materials/reine_character_guide.md`](research_evidence_prep/dataset_materials/reine_character_guide.md) - sanitized character guide for dataset creation.
- [`research_evidence_prep/dataset_materials/identity_stress_test_prompts.txt`](research_evidence_prep/dataset_materials/identity_stress_test_prompts.txt) - 15-prompt identity stress test.

## Current Best Run from the Dump

The current thesis draft treats **Lower-5+CoT** as the strongest ReINE configuration:

- layers: `0-4`
- CoT trace in training data: enabled
- trainable parameters: `409,610`
- zero-shot identity stress-test score: `30/30`

In these notes, **CoT enabled/disabled refers to whether the adapter was trained with chain-of-thought-style trace material in the supervised examples**. It does not mean the evaluation forcibly allowed or blocked the model from emitting visible CoT at inference time. Output visibility is still governed by the model, tokenizer/chat template, prompting, and evaluation script behavior.

The standard LoRA baseline reaches `22/30` with `33,030,144` trainable parameters in the same thesis comparison.

## Recreating the Experiments

The exact thesis runs depend on private/local synthetic JSONL datasets. The raw datasets are not included in the prep folder yet because some rows need sanitation before public release. To recreate the experiments, prepare reviewed JSONL datasets with the same schema and update the config paths.

Expected dataset row format for ReINE training:

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

The LoRA runner also accepts simple prompt/response style rows:

```json
{"prompt": "...", "response": "..."}
{"instruction": "...", "output": "..."}
```

### 1. Environment

Recommended environment:

- Linux or WSL with CUDA-capable GPU
- Python 3.11
- PyTorch with CUDA
- Hugging Face access for `unsloth/Qwen3-4B-Thinking-2507`

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If `unsloth` needs a platform-specific install command, follow the current Unsloth install guide for your CUDA/PyTorch version, then rerun the install.

### 2. Prepare Input Files

Useful prompt/material files:

- `test_prompt.txt` - 15-prompt identity stress test.
- `instruction_prompt.txt` - ReINE identity/persona instruction anchor.
- `research_evidence_prep/dataset_materials/reine_character_guide.md` - sanitized dataset character guide.

For exact follow-up experiments, place or configure a reviewed 665-example dataset equivalent to:

```text
tunedataset_600v2.jsonl
```

For the initial small ablation, use the compact 157-example dataset equivalent to:

```text
tunedataset.jsonl
```

### 3. Train One ReINE Adapter

Basic Lower-5+CoT-style training command:

```bash
python train_adapter.py \
  --config configlow.yaml \
  --save_dir runs/lower5_cot \
  --include_think true \
  --seed 42
```

Notes:

- `configlow.yaml` is the Lower-5-style config from the dump.
- `training.include_think` inside the YAML overrides the CLI flag if present.
- `include_think` means the supervised training examples include the reasoning trace/answer-boundary material used by the run. It is a training-data condition, not a guarantee that inference will or will not display chain-of-thought text.
- Main outputs are `adapter_final.pt`, `run_metadata.json`, `resolved_config.yaml`, and checkpoint files.

### 4. Run Identity Ablation Evaluation

Zero-shot/stateless evaluation:

```bash
python scripts/ablation.py \
  --ckpt runs/lower5_cot/adapter_final.pt \
  --config runs/lower5_cot/resolved_config.yaml \
  --prompts test_prompt.txt \
  --outdir runs/lower5_cot/eval_zs \
  --tag lower5_cot_zs
```

Carry-history evaluation:

```bash
python scripts/ablation.py \
  --ckpt runs/lower5_cot/adapter_final.pt \
  --config runs/lower5_cot/resolved_config.yaml \
  --prompts test_prompt.txt \
  --outdir runs/lower5_cot/eval_history \
  --tag lower5_cot_history \
  --carry_history
```

Each evaluation writes raw JSONL, clean JSONL, and metadata JSON.

### 5. Recreate the Fixed-Variant Suite

This reproduces the lower/middle/upper/full/default family used for the 157-example and 665-example fixed-variant comparisons:

```bash
python scripts/reine_fixed_pipeline.py \
  --base-config config.yaml \
  --train-script train_adapter.py \
  --ablation-script scripts/ablation.py \
  --prompts test_prompt.txt \
  --outdir runs/fixed_variants \
  --seed 42
```

For the 665-example run, use a config whose dataset path points to the reviewed `tunedataset_600v2.jsonl` equivalent:

```bash
python scripts/reine_fixed_pipeline.py \
  --base-config config600v2.yaml \
  --train-script train_adapter.py \
  --ablation-script scripts/ablation.py \
  --prompts test_prompt.txt \
  --outdir runs/600_reine \
  --seed 42
```

Expected variant folders:

- `lower_third`
- `middle_third`
- `upper_third`
- `full_symmetric`
- `default`

### 6. Recreate the LoRA Baseline

The thesis comparator is standard LoRA, not QLoRA:

```bash
python scripts/train_lora_unsloth_reine.py \
  --config reine_lora_default_config.yaml \
  --dataset_path tunedataset_600v2.jsonl \
  --prompt_txt test_prompt.txt \
  --save_dir runs/reine_lora_baseline
```

Main outputs:

- `adapter_final/`
- `run_metadata.json`
- `resolved_config.yaml`
- `training_log.jsonl`
- `answers/base_before_training.jsonl`
- `answers/lora_after_training.jsonl`
- `answers/comparison.md`

### 7. What to Compare

For each run, keep:

- generated config and resolved config,
- `run_metadata.json`,
- training logs,
- raw and clean evaluation JSONL,
- evaluation metadata,
- human scoring sheet or notes.

Do not commit normal Git copies of:

- `.pt` checkpoints,
- `.safetensors`,
- raw `.zip` archives,
- `tokenizer.json`,
- repeated extracted run folders.

## Release Caution

The raw training JSONL datasets are not included in the prep folder yet. Some rows still contain raw persona-lore material that should be sanitized before public release. Model checkpoints, `.pt`, `.safetensors`, raw zips, and tokenizer artifacts should also stay out of normal Git history unless intentionally published through Git LFS or releases.

## Scope

This is proof-of-concept research evidence for identity/persona steering on one host model and one targeted stress-test setup. It should not be read as a broad benchmark claim or a general LoRA comparison.