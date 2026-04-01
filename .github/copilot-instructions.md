<!-- Copilot instructions for AI coding agents working on this repo -->
# Quick orientation for AI coding agents

This repo contains two CLIP-style pipelines wired to a shared core. Be concise, make minimal changes, and prefer to update existing configs and small helpers over large refactors.

- **Big picture:** shared utilities live in `src/clip_core/` and two project folders implement pipelines:
  - `projects/kiba_dti/` — protein–ligand retrieval on KIBA
  - `projects/mibig_bgc_np/` — BGC–compound retrieval on MIBiG
  See [README.md](../README.md) for full workflows.

- **Data / cache / results flow:**
  - Preprocess raw data (MIBiG): `scripts/preprocess_mibig.py` → `data/MIBIG/processed`
  - Cache base features: project `cache_features` script writes `cache/.../{*_features.pt,cache_index.json,ligand_fingerprints.pt,protein_embeddings.pt}` (see `cache/` examples)
  - Train contrastive: `python -m projects.<proj>.scripts.train_contrastive --data_dir ... --cache_dir ... --config projects/<proj>/configs/default.yaml`
  - Evaluate / viz / downstream use `eval_retrieval`, `viz_umap`, `train_downstream` entrypoints; outputs go under `results/.../`

- **Important file examples:**
  - Project configs: [projects/kiba_dti/configs/default.yaml](../projects/kiba_dti/configs/default.yaml) and [projects/mibig_bgc_np/configs/default.yaml](../projects/mibig_bgc_np/configs/default.yaml)
  - Shared code and helpers: [src/clip_core/](../src/clip_core/)
  - Top-level README with canonical CLI examples: [README.md](../README.md)

- **Conventions & patterns to follow**
  - CLI entrypoints are always invoked as modules: `python -m projects.<proj>.scripts.<name>` — prefer this style when adding examples or tests.
  - Feature caching is explicit and reused across runs. Do not remove or rename cache keys without updating cache writers/readers (`cache_index.json` is authoritative).
  - Configs are YAML; prefer adding flags to existing configs rather than adding many CLI args. Use the project's `configs/default.yaml` as canonical defaults.
  - Model encoders: ESM2 (sequence) and Morgan fingerprints (molecules) are standard; see `featurization` keys in the configs above.

- **Dependencies & environment**
  - Python >= 3.10. Core deps listed in [pyproject.toml](../pyproject.toml) (PyTorch, transformers, rdkit, pandas, umap, etc.). Use `pip install -e .` in a venv for development.

- **Testing & quick checks**
  - Unit tests live in `tests/` and the repo uses `pytest` (see `pyproject.toml`). Run `pytest -q` from the repo root.
  - Small behavioral changes: run the cached pipeline steps locally on a tiny subset (create minimal `data` with 1–2 rows) to validate I/O shapes.

- **When editing code**
  - Keep public CLI and cache file formats stable. If you must change a format, update the writer + reader + README examples.
  - Prefer small, incremental PRs that touch config, cache handling, or a single script. Add tests for new behavior in `tests/`.

If any of these points are ambiguous or you want a concrete example (small end-to-end run, config diff, or a test), tell me which part and I will expand with a concrete code change or example command.
