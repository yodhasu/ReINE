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
- CoT supervision: enabled
- trainable parameters: `409,610`
- zero-shot identity stress-test score: `30/30`

The standard LoRA baseline reaches `22/30` with `33,030,144` trainable parameters in the same thesis comparison.

## Release Caution

The raw training JSONL datasets are not included in the prep folder yet. Some rows still contain raw persona-lore material that should be sanitized before public release. Model checkpoints, `.pt`, `.safetensors`, raw zips, and tokenizer artifacts should also stay out of normal Git history unless intentionally published through Git LFS or releases.

## Scope

This is proof-of-concept research evidence for identity/persona steering on one host model and one targeted stress-test setup. It should not be read as a broad benchmark claim or a general LoRA comparison.