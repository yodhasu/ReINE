# Dataset Availability Note

The original synthetic training datasets are not included in this public-prep folder yet.

Reason: during cleanup, the source datasets were found to contain raw persona-lore material that should be reviewed and sanitized before public release. The experiment metadata, configs, prompts, evaluation outputs, and run notes are included so the research trail remains understandable without publishing raw dataset rows prematurely.

Original dataset names referenced by configs and metadata:

- `tunedataset.jsonl` / compact 157-example dataset
- `tunedataset_600.jsonl`
- `tunedataset_600v2.jsonl` / 665-example dataset used by the later ReINE and LoRA runs

Before release, create a reviewed dataset package that preserves the experimental meaning while removing private jokes, sexualized avatar details, and any material unsuitable for a research repository.
