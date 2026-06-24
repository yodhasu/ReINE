# ReINE

ReINE, short for **Residual Information Network Editing**, is a frozen-host persona steering experiment for causal language models. The host model stays frozen while small trainable residual MicroAdapters are attached to selected transformer layers through forward hooks.

This repository is organized as both a code release and a research evidence bundle. The main implementation files are kept at the front of the repo for readability:

- `adapter.py` - residual MicroAdapter wrapper and hidden-state intervention logic.
- `train_adapter.py` - ReINE adapter training script.
- `chat_adapter.py` - interactive inference script using a trained adapter.

Additional scripts live in `scripts/`:

- `ablation.py` - identity stress-test evaluator for ReINE checkpoints.
- `reine_fixed_pipeline.py` - fixed lower/middle/upper/full/default variant runner.
- `reine_auto_pipeline.py` - exploratory pipeline for variant search and scoring.
- `train_lora_unsloth_reine.py` - standard LoRA baseline runner.
- `legacy_qlora/` - older QLoRA experiments kept for provenance.

## Research Materials

- `configs/` contains ReINE and LoRA configs used across the experiments.
- `prompts/` contains the identity stress-test prompts and identity instruction prompt.
- `dataset_materials/` contains dataset construction material, including the sanitized ReINE character guide.
- `data/` documents the original synthetic identity datasets, which are withheld from this public-prep folder until raw persona-lore rows are reviewed and sanitized.
- `research_evidence/` contains lightweight experiment evidence: manifests, configs, metadata, logs, evaluation JSONL, and comparison notes.
- `result_notes.md` is a detailed rough research notebook that records experiment families and caveats that did not fit cleanly into the thesis draft.

Large model files and checkpoints are intentionally not included in normal Git history. The repository keeps evidence and reproducibility traces, but excludes `.pt`, `.safetensors`, raw zip archives, and large tokenizer/model artifacts.

The original raw training JSONL files are also withheld from this prep folder for now because some rows contain raw character-lore material that needs cleanup before public release.

## Experiment Families

The evidence is grouped into:

- `initial_157_fixed_variants` - April 30 fixed depth-allocation suite on the 157-example dataset.
- `followup_665_fixed_variants` - May 3 fixed variants on the 665-example dataset.
- `lower_focused_665_probes` - lower-layer probes including layer 0, Lower-5, Lower-5+CoT, and 11-1-1.
- `lora_baseline_665` - standard LoRA comparator trained on the same 665-example dataset.
- `legacy_ablation_probes` - older March 31 qualitative ablation outputs.

The current thesis draft treats **Lower-5+CoT** as the strongest ReINE configuration in this dump: layers `0-4`, trained with CoT trace material, 409,610 trainable parameters, and 30/30 zero-shot identity accuracy on the 15-prompt stress test.

Terminology clarification: **CoT enabled/disabled in these notes describes the training data condition**. A CoT-enabled run used supervised examples with chain-of-thought-style trace material and answer-boundary behavior; a CoT-disabled run did not. This label does not mean the evaluation explicitly permitted or prohibited the model from emitting visible CoT during inference.

## Scope

This is a proof-of-concept research repository. The results are limited to one host model, synthetic identity-steering datasets, and a targeted identity stress test. The repo should be read as experimental evidence for residual activation-space persona steering, not as a broad benchmark claim.
