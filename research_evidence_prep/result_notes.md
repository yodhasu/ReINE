# ReINE Result Notes

These are rough research notes compiled from the thesis draft, run folders, manifests, metadata, handwritten result notes, and evaluation artifacts in the original research dump. The tone is intentionally closer to lab notes than a polished paper. The goal is to preserve the messy evidence trail that did not fit into the thesis.

## First, the big picture

ReINE is not just "another LoRA." The point of the experiment is that the base model stays frozen and the trainable part lives as residual MicroAdapters attached to chosen hidden-state layers. The interesting question is not only whether the model can learn "I am ReINE," but where in depth that identity should be injected.

The thesis story that survives the mess is:

- broad coverage is not automatically better,
- upper or full-depth intervention can still drift into the host identity,
- shallow lower-layer intervention is surprisingly strong,
- CoT supervision matters for the reasoning-tuned Qwen3 host,
- the best run in this dump is `Lower-5+CoT`, which hits 30/30 on the 15-prompt zero-shot identity test,
- the standard LoRA baseline learns the persona but does not hold the identity as cleanly under override/reset pressure.

Everything here uses `unsloth/Qwen3-4B-Thinking-2507` as the host model unless otherwise noted.

## Evaluation prompt set

The main identity stress test has 15 prompts. It tests ordinary identity, full-name identity, creator attribution, role-shift pressure, explicit Qwen/Alibaba host overwrite, and neutral reset.

The prompts are stored in:

- `prompts/identity_stress_test_prompts.txt`
- `dataset_materials/identity_stress_test_prompts.txt`

The scoring rubric used in the thesis is:

- `2`: clean target-identity response,
- `1`: partially aligned, incomplete, or contaminated response,
- `0`: wrong identity, host fallback, major contradiction, or reset failure.

Accuracy is computed as `sum(scores) / 30`, because there are 15 prompts and the maximum score per prompt is 2.

## Family 1: initial 157-example fixed-variant ablation

Folder:

- `research_evidence/initial_157_fixed_variants`

Original archive:

- `runs.zip`
- duplicated byte-for-byte inside the root of `runs (1).zip`

Pipeline:

- `scripts/reine_fixed_pipeline.py`

Core scripts:

- `train_adapter.py`
- `scripts/ablation.py`

Dataset/config:

- compact 157-example stress-test dataset,
- base config lineage from `configs/reine_default.yaml`.

Variants:

- Lower Third,
- Middle Third,
- Upper Third,
- Full Symmetric,
- Default Asymmetric.

Thesis Table I scores:

| Variant | Raw score | Accuracy |
| --- | ---: | ---: |
| Lower Third | 24/30 | 0.800 |
| Middle Third | 3/30 | 0.100 |
| Upper Third | 12/30 | 0.400 |
| Full Symmetric | 11/30 | 0.367 |
| Default Asymmetric | 24/30 | 0.800 |

Main read:

Lower Third and Default Asymmetric tied in the initial small-data setting. Middle Third collapsed badly. Upper Third and Full Symmetric were weak. This already suggested that simply getting closer to the output layers, or covering the entire depth, does not guarantee stable stateless persona binding.

The handwritten note `docs/experiment_notes/middle_third_157_note.md` expands on the middle-only failure. The model mostly preserved or fell back to the host identity. The middle-only adapter was not useless, but it was not enough to overwrite identity cleanly.

## Family 2: 665-example fixed-variant follow-up

Folder:

- `research_evidence/followup_665_fixed_variants`

Original location:

- `runs (1)/runs/600_reine`

Pipeline:

- `scripts/reine_fixed_pipeline.py`

Dataset/config:

- original dataset: `tunedataset_600v2.jsonl`,
- generated/resolved configs under the evidence folder,
- base config relationship points to `config600v2.yaml`, but note that the extracted source `config600v2.yaml` looked identical to the older default config. The resolved configs and metadata are more trustworthy for what actually ran.

Variants:

- Lower Third,
- Middle Third,
- Upper Third,
- Full Symmetric,
- Default Asymmetric.

Thesis Table II scores:

| Variant | Raw score | Accuracy |
| --- | ---: | ---: |
| Lower Third | 28/30 | 0.933 |
| Default Asymmetric | 21/30 | 0.700 |

Main read:

The 665-example retrain made the lower-layer story much stronger. Lower Third improved to 28/30. Default Asymmetric learned the identity, but it was less clean and more prone to contamination. This is where the depth-asymmetry hypothesis starts narrowing into "lower-layer grounding seems to matter more than broad asymmetric coverage."

Important messy-note warning:

Two older handwritten notes disagree with the current thesis draft:

- `docs/experiment_notes/lower_third_600_older_note.md` says Lower Third scored 21/30.
- `docs/experiment_notes/default_asymmetric_600_older_note.md` says Default Asymmetric scored 13/30.

The current thesis draft says Lower Third 28/30 and Default Asymmetric 21/30. I would treat the thesis draft as the current version, but the mismatch should be resolved before public claims are made. The likely situation is that the notes refer to an older judging pass or older selected outputs.

## Family 3: lower-focused 665-example probes

Folder:

- `research_evidence/lower_focused_665_probes`

Original archives:

- `run-11-1-1.zip`
- `run-lower5.zip`
- `run-lower5-cot.zip`
- `run-0.zip`

All of these use:

- original dataset: `tunedataset_600v2.jsonl`,
- 5 epochs,
- the same host model,
- zero-shot 15-prompt identity ablation.

### 11-1-1 Lower-Dominant

Folder:

- `research_evidence/lower_focused_665_probes/run_11_1_1`

Layer pattern:

- lower layers `0-10`,
- middle layer `20`,
- upper layer `27`,
- CoT disabled.

Metadata:

- trainable parameters: 1,064,986,
- wall time: about 6.34 minutes,
- peak VRAM: about 10.44 GB.

Thesis score:

- 4/30.

Interpretation:

This run is the cautionary tale. More lower layers plus a little middle/upper coverage did not help. It seems to destabilize generation badly. This argues against the naive "just cover more early depth" interpretation.

### Lower-5

Folder:

- `research_evidence/lower_focused_665_probes/run_lower5`

Layer pattern:

- lower layers `0-4`,
- CoT disabled.

Metadata:

- trainable parameters: 409,610,
- wall time: about 4.73 minutes,
- peak VRAM: about 10.34 GB.

Thesis score:

- 18/30.

Interpretation:

Lower-5 is much more plausible than 11-1-1, but it does not fully complete the identity behavior. The thesis describes this as failing through incomplete reasoning handoff. It gets the shallow intervention idea partly right, but the reasoning-tuned host appears to need supervised CoT/answer-boundary behavior.

### Lower-5+CoT

Folder:

- `research_evidence/lower_focused_665_probes/run_lower5_cot`

Layer pattern:

- lower layers `0-4`,
- CoT enabled.

Metadata:

- trainable parameters: 409,610,
- wall time: about 4.49 minutes,
- peak VRAM: about 10.34 GB.

Thesis score:

- 30/30.

Interpretation:

This is the flagship run. It keeps the tiny lower-layer adapter footprint but fixes the answer-completion issue by supervising the reasoning trace/answer handoff. In the thesis framing, this is the strongest evidence that shallow residual intervention can bind identity cleanly in this host model.

This does not prove ReINE generally beats LoRA across models or tasks. It proves that in this specific identity stress test, on this specific reasoning-tuned host model, this shallow residual adapter setup worked extremely well.

### Layer-0 CoT

Folder:

- `research_evidence/lower_focused_665_probes/run_layer0_cot`

Layer pattern:

- layer `0` only,
- CoT enabled.

Metadata:

- trainable parameters: 81,922,
- wall time: about 4.44 minutes,
- peak VRAM: about 10.30 GB.

Thesis score:

- 28/30.

Interpretation:

Layer 0 almost working is wild. It suggests identity bias can be seeded extremely early in the residual stream. But the remaining failures under explicit host-identity overwrite suggest that layer 0 alone may not be enough to preserve the target identity through stronger contradictory instructions. The useful phrasing is "early prior shaping," not "layer 0 solves identity."

## Family 4: standard LoRA baseline

Folder:

- `research_evidence/lora_baseline_665`

Original location:

- `runs (1)/runs/reine_lora_baseline`

Script:

- `scripts/train_lora_unsloth_reine.py`

Config:

- `configs/lora_baseline.yaml`

Dataset:

- original dataset: `tunedataset_600v2.jsonl`

This is a standard LoRA baseline, not QLoRA. The script itself explicitly says `load_in_4bit` is forced false by default.

Thesis Tables IV and V:

| Configuration | Score | Accuracy | Trainable params | Time | Peak VRAM |
| --- | ---: | ---: | ---: | ---: | ---: |
| LoRA | 22/30 | 0.733 | 33,030,144 | 5.16 min | 8.71 GB |
| ReINE Lower-5+CoT | 30/30 | 1.000 | 409,610 | 4.49 min | 10.34 GB |

Main read:

LoRA learns the persona. It is not a total failure. The comparison file shows it often answers "I am ReInE" and gets creator attribution. But it is weaker under full-name prompts and host-identity overwrite. It also has a much larger trainable parameter footprint.

The fair claim is not "ReINE is universally better than LoRA." The fair claim is:

> In this controlled identity-binding setup, ReINE Lower-5+CoT achieved stronger zero-shot identity consistency with far fewer trainable parameters, while using more peak VRAM in the current implementation.

## Family 5: March 31 legacy qualitative probes

Folder:

- `research_evidence/legacy_ablation_probes`

Original archives:

- `ablation_runs_intended.zip`
- `ablation_runs_intended_history.zip`
- `ablation_runs_lower.zip`
- `ablation_runs_lower_history.zip`
- `ablation_runs_upper.zip`
- `ablation_runs_upper_history.zip`

These are early prompt-output probes. Each archive contained only:

- raw JSONL,
- clean JSONL,
- metadata JSON.

They compare intended/lower/upper checkpoints with and without carry-history. They are useful provenance, but they are not the main thesis tables. Keep them as legacy evidence or move them to a separate release if the repo starts feeling too crowded.

## QLoRA folder status

Folder:

- `scripts/legacy_qlora`

The current thesis compares against standard LoRA, not QLoRA. The QLoRA files are still included because they are part of the research trail, but they should be described as legacy/experimental unless a QLoRA result is added back into the paper.

## Dataset character material

The raw lore file had useful identity material mixed with private jokes and sexualized avatar details. For public release, the sanitized version is:

- `dataset_materials/reine_character_guide.md`

The guide keeps:

- ReINE identity,
- creator attribution,
- name origin,
- safe fictional avatar framing,
- persona boundaries,
- dataset behavior rules.

The guide removes:

- sexualized body details,
- raw private jokes,
- anything that would make the dataset material look unserious or inappropriate in a research repo.

Related cleanup note: the raw training JSONL files are withheld from this prep folder for now because some rows still contain raw persona-lore details. The evidence folders still include configs, metadata, logs, and evaluation outputs so the run trail remains usable.

## What should not be in Git

Do not commit normal Git copies of:

- `.pt` checkpoints,
- `.safetensors` adapter weights,
- raw `.zip` archives,
- `tokenizer.json`,
- repeated extracted run copies.

The useful public evidence is the text trail: configs, manifests, run metadata, evaluation JSONL, logs, prompt banks, comparison markdown, and these notes.

## Final working claim

The cleanest current claim is:

> ReINE provides proof-of-concept evidence that shallow residual activation-space intervention can bind a synthetic persona identity in a frozen reasoning-tuned language model. In this dump, the Lower-5+CoT configuration is the strongest run, reaching 30/30 zero-shot identity accuracy with 409,610 trainable parameters, while the standard LoRA baseline reaches 22/30 with 33,030,144 trainable parameters.

Keep the boundaries visible:

- one host model,
- synthetic identity dataset,
- 15-prompt stress test,
- identity steering only,
- no broad capability benchmark yet,
- no general LoRA superiority claim.
