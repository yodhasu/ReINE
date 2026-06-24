# ReINE Result Notes

The detailed chaotic-but-scientific run notes are here:

- [`research_evidence_prep/result_notes.md`](research_evidence_prep/result_notes.md)

That file compiles the experiment families from the thesis draft, run folders, manifests, metadata, handwritten notes, and evaluation artifacts.

Quick map:

- `runs.zip` - initial April 30 fixed-variant suite on the 157-example dataset.
- `runs (1).zip` - superset containing the same initial suite plus `600_reine` and `reine_lora_baseline`.
- `run-11-1-1.zip` - lower-dominant probe, layers `0-10, 20, 27`, no CoT, thesis score `4/30`.
- `run-lower5.zip` - lower layers `0-4`, no CoT, thesis score `18/30`.
- `run-lower5-cot.zip` - lower layers `0-4`, CoT enabled, thesis score `30/30`.
- `run-0.zip` - layer `0` only, CoT enabled, thesis score `28/30`.
- `reine_lora_baseline` - standard LoRA baseline, thesis score `22/30`.

The raw training JSONL datasets are intentionally withheld from the prep commit until sanitized, because some rows contain raw persona-lore details that should not be published as-is.