# Publication cleanup map

Do not delete intermediate artifacts until the paper tables are locked and each
reported number has a provenance entry in `EXPERIMENTS.md`.

## Keep in the public repository

- reusable source under `projects/` and `src/`;
- compact reproducibility configs;
- tests;
- curated experiment descriptions under `experiments/`;
- final paper tables or machine-readable summaries selected for release;
- `EXPERIMENTS.md`, edited to remove private absolute paths if necessary.

## Local/intermediate artifacts

- `results/intermediate/`: screening runs, failed runs, pilot folds, checkpoints;
- `cache/`: downloaded or computed embeddings and token-level inputs;
- `condor/`: Saarland-specific submit files and absolute paths;
- cluster scheduler log directories: local HTCondor/DAGMan stdout/stderr files.

These directories are already ignored by git. Preserve them locally until final
model selection, then publish only the necessary generation instructions and
small derived tables.
