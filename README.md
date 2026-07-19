# ReINE (Residual Information Network Editing)

> **Quick Summary**: ReINE steers frozen LLMs by attaching tiny trainable "MicroAdapters" to selected transformer layers. No host weights are changed. Just residual hooks + a small adapter.

## 👋 Author's Note (Read This First)

<!-- This repo contains my undergraduate thesis work (BINUS University, Global Class). The paper was accepted to **ICIMTECH 2026**, but the story doesn't end there.

**After defense**, I stress-tested my own results with 48 hours of A40 rentals and found:
- The "best" config from the paper (`Lower-5+CoT`) actually **degrades math/reasoning** because the synthetic CoT data was low-quality.
- The *actual* best config is **`5-each-nothinkhice`**: full-depth adapters + anchor loss + **unsupervised CoT** (`include_think=False`).
- KL divergence on middle layers? Toxic. Anchor loss? Critical for preserving style.

I documented the full post-defense journey, raw logs, and "aha!" moments in my **[Research Diary](LINK_TO_NOTION_OR_MD)**. If you're here to reproduce the paper, skip to "Recreating Experiments". If you're here to understand *why* things broke and how we fixed them, start with the diary.

No formalities. Just honest research.

— Yodha -->

This repo contains my undergraduate thesis work (BINUS University, Global Class). The paper was accepted to **ICIMTECH 2026** and I already finished my defense, but I just don't have the patience to wait someone debunk my work so I make **After Defense Research Diary** which is basically just documented runs of my experiment trying to prove I'm wrong. It answers some limitations listed in paper and actually debunk some hypothesis mentioned in my paper.

I documented the workspace 1 to 1 inside `post-thesis-defense-experiments/` and the research diary here: https://app.notion.com/p/ReINE-Research-Diary-3a2a0a92519c806481a3dc4f3cf95c46?source=copy_link

-- Yodha

---

## 📦 What's in This Repo

```
ReINE/
├── adapter.py                  # [LATEST] Core MicroAdapter + DeepAdapterWrapper logic
├── train_adapter.py            # [LATEST] Depth-asymmetric training script (includes post-defense ablation fixes)
├── chat_adapter.py             # [LATEST] Inference script
├── tunedataset_600v2           # Exact same dataset used in paper
├── instruction_prompt          # Persona anchor text used for latent alignment (same thing used in paper)
├── requirements                # Python dependencies
├── LICENSE                     # MIT License
├── paper-related-proofs/       # [ARCHIVE] Exact state of the repository during Thesis submission
│   ├── result/                 # Result output from thesis experiments
│   ├── scripts/                # Legacy helper scripts
│   ├── test-runs/              # Legacy test runs
│   ├── adapter.py              # [OLD] Version used in the paper, doesn't have dedicated remove_hooks()
│   ├── train_adapter.py        # [OLD] Version used in the paper, have different loss formula
│   ├── config*.yaml            # [OLD] Configs used in the paper (lower5, 600v2, etc.)
│   ├── RESULT_NOTES.md         # Original rough lab notes & result summaries from the thesis
│   └── tunedataset_*.jsonl     # [OLD] Historical dataset versions
│
└── post-thesis-defense-experiments/
    ├── workspace18-07-26/      # [ARTIFACTS] Extracted workspace from experiments (KL-mid, NoAnchors, etc.)
    ── workspace19-07-26/      # [ARTIFACTS] Extracted workspace from experiments (NoThinkHiCE discovery)
    another workspace coming soon...
```

You can actually just clone a workspace inside the post-thesis-defense-experiments if you wanna reproduce the results easily.

---

## 🚀 Recreating Experiments (Quick Start)

### 1. Environment
```bash
# Linux/WSL with CUDA GPU recommended
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# If Unsloth needs platform-specific install:
# Follow https://docs.unsloth.ai/ for your CUDA/PyTorch version
# If you encounter torch version problem please install the latest torch version OR uninstall torchao
```

### 2. Prepare Data
The repo includes **sanitized** prompt materials:
- `test_prompt.txt` – 15-prompt identity stress test
- `instruction_prompt.txt` – ReINE persona anchor text
- `research_evidence_prep/dataset_materials/` – Character guide, prompt templates

**Raw training JSONL files are included** To reproduce exact results. However if you wanna use it for personal use I highly encouraged to make your own dataset.

**Dataset row format**:
```json
{"messages": [
  {"role": "user", "content": "Who are you?"},
  {"role": "assistant", "content": "<think>...</think>I am ReInE."}
]}
```

### 3. Train a ReINE Adapter
**General use**:
```bash
python train_adapter.py \
  --config config_filename.yaml \
  --save_dir foldername/tosaveruns \
  --include_think [boolean] \
  --seed int
```

<!-- ### 4. Evaluate
**Zero-shot identity stress test**:
```bash
python scripts/ablation.py \
  --ckpt runs/5-each-nothinkhice/adapter_final.pt \
  --config runs/5-each-nothinkhice/resolved_config.yaml \
  --prompts test_prompt.txt \
  --outdir runs/5-each-nothinkhice/eval \
  --tag nothinkhice_zs
```

**Math/reasoning evaluation** (optional but recommended):
```bash
python scripts/eval_math.py \
  --ckpt runs/5-each-nothinkhice/adapter_final.pt \
  --config runs/5-each-nothinkhice/resolved_config.yaml \
  --prompts math_prompts.txt \
  --outdir runs/5-each-nothinkhice/eval_math
```

### 5. Reproduce the Full Ablation Suite
```bash
python scripts/reine_fixed_pipeline.py \
  --base-config config.yaml \
  --train-script train_adapter.py \
  --ablation-script scripts/ablation.py \
  --prompts test_prompt.txt \
  --outdir runs/fixed_variants \
  --seed 42
```

Expected variant folders: `lower_third`, `middle_third`, `upper_third`, `full_symmetric`, `default`.

---

## ⚠️ Important Warnings

### 1. CoT Supervision Degrades Reasoning
If you set `include_think: true` with synthetic CoT data:
- ✅ Identity binding may improve (30/30)
- ❌ Math/reasoning capabilities will degrade (see `5-each-runs` logs)
- ❌ Model may generate "meta-compliance" text instead of actual reasoning

**Recommendation**: Use `include_think: false` + heavy `think_end_weight` (5.0) for thinking models like Qwen3-4B-Thinking.

### 2. KL Divergence on Middle Layers = Toxic
Adding `middle_kl_weight > 0` causes semantic overwriting:
- Model treats math prompts as identity queries
- Outputs become confused, repetitive, or nonsensical
- Final KL loss spikes (575+ vs ~387 baseline)

**Recommendation**: Keep `*_kl_weight: 0.0` for persona steering tasks.

### 3. Anchor Loss Preserves Styling + Reasoning
Removing anchor loss (`*_anchor_weight: 0`):
- ✅ Identity binding still works
- ❌ Math accuracy drops subtly
- ❌ Responses become stylistically "flat"

**Recommendation**: Keep `lower_anchor_weight: 1.0` (cosine) to tether adapted layers to the host's linguistic manifold.

---

## 🔗 Links & Citation

- **Paper PDF**: [`paper-related-proofs/Paper_REINE_Final.pdf`](paper-related-proofs/Paper_REINE_Final.pdf)
- **ICIMTECH Slides**: [`paper-related-proofs/presentation.pdf`](paper-related-proofs/presentation.pdf)
- **Research Diary**: [LINK_TO_NOTION_OR_MD] ← Raw thoughts, debugging logs, and post-defense discoveries
- **Reproducibility Guide**: [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) ← Exact commands, expected outputs, troubleshooting

```bibtex
@inproceedings{pratama2026reine,
  title={Residual Information Network Editing for Persona Steering in Frozen Language Models},
  author={Pratama, Alethea Agung Yodha and Elhan, Azka Dhafin and Kliveson, Collin and Hidayaturrahman},
  booktitle={International Conference on Information Management and Technology (ICIMTECH)},
  year={2026}
}
``` -->

---

## 📜 License & Release Caution

- Code: MIT License
- Paper materials: CC-BY 4.0\
- Model checkpoints (`.pt`, `.safetensors`): Exclude from normal Git history; use Git LFS or releases if publishing.

> This is proof-of-concept research evidence for identity/persona steering on one host model (`unsloth/Qwen3-4B-Thinking-2507`) and one targeted stress-test setup. It should not be read as a broad benchmark claim or a general LoRA comparison.
```
