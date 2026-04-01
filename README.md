# BGC2NP-CLIP: Contrastive Cross-Modal Retrieval Between Biosynthetic Gene Clusters and Natural Products

This repo contains two separate CLIP-style pipelines built on shared utilities:

- `projects/mibig_bgc_np/`: BGC-compound retrieval on MIBiG
- `projects/kiba_dti/`: protein-ligand retrieval on KIBA
- `src/clip_core/`: shared config, logging, losses, retrieval, and caching utilities

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

Optional, only if you want parquet output from visualization scripts:

```bash
pip install pyarrow
```

## Repository layout

```text
src/clip_core/                  Shared losses, retrieval, config, logging, cache
src/kiba_clip/                  KIBA package wired to clip_core
src/mibig_clip/                 MIBiG shared package code
projects/kiba_dti/              KIBA configs and script entrypoints
projects/mibig_bgc_np/          MIBiG configs and script entrypoints
```

## MIBiG workflow

This is the BGC-compound pipeline.

The implemented baseline is:

- per-protein encoding
- mean pooling over proteins inside each BGC
- Morgan fingerprints for compounds
- CLIP-style dual projection heads trained on cached features
- multi-positive retrieval in both directions

### What the MIBiG data should look like

After preprocessing, the pipeline expects:

- `data/MIBIG/processed/mibig_pairs.tsv`
- `data/MIBIG/processed/bgc_proteins.jsonl`
- `data/MIBIG/splits/random_seed42.tsv` or another split TSV

`mibig_pairs.tsv` should contain at least:

- `bgc_id`
- compound identifier column, usually `smiles`
- `bgc_class` if you want to run the downstream BGC class classifier
- one split source, either:
  - `split` directly in the table, or
  - a separate split TSV keyed by `bgc_id`

`bgc_proteins.jsonl` should map each `bgc_id` to:

- `protein_ids`
- `protein_seqs`

### Step 0: Preprocess raw MIBiG data

Purpose:

- parse the MIBiG FASTA and JSON metadata
- build BGC-compound pair rows
- build BGC-level train/val/test splits

Run:

```bash
python scripts/preprocess_mibig.py \
  --fasta_path data/MIBIG/mibig_prot_seqs_4.0.fasta \
  --json_dir data/MIBIG/mibig_json_4.0 \
  --out_dir data/MIBIG/processed \
  --splits_dir data/MIBIG/splits \
  --seed 42 \
  --make_splits both \
  --cold_k 5 \
  --cold_threshold 0.3
```

Writes:

- `data/MIBIG/processed/mibig_pairs.tsv`
- `data/MIBIG/processed/bgc_proteins.jsonl`
- `data/MIBIG/processed/mibig_summary.json`
- `data/MIBIG/splits/random_seed42.tsv`
- `data/MIBIG/splits/cold_seed42_k5_thr0.3.tsv`
- `data/MIBIG/splits/cold_seed42_k5_thr0.3_report.json`

Use this step only when starting from the raw MIBiG FASTA/JSON files.

### Step 1: Cache BGC and compound features

Purpose:

- encode each protein sequence
- mean-pool protein embeddings within each BGC
- compute Morgan fingerprints for compounds
- save reusable cached features

Run:

```bash
python -m projects.mibig_bgc_np.scripts.cache_features \
  --data_dir data/MIBIG/processed \
  --out_dir cache/mibig_default \
  --config projects/mibig_bgc_np/configs/default.yaml
```

Writes:

- `cache/mibig_default/bgc_features.pt`
- `cache/mibig_default/compound_features.pt`
- `cache/mibig_default/cache_index.json`

By default the config uses:

- `one_hot` for BGC protein encoding
- `morgan` for compound encoding

### Step 2: Train the contrastive model

Purpose:

- train projection heads on cached BGC and compound features
- select the best checkpoint using validation mean MRR

Run:

```bash
python -m projects.mibig_bgc_np.scripts.train_contrastive \
  --data_dir data/MIBIG/processed \
  --cache_dir cache/mibig_default \
  --config projects/mibig_bgc_np/configs/default.yaml
```

If you want a different split file:

```bash
python -m projects.mibig_bgc_np.scripts.train_contrastive \
  --data_dir data/MIBIG/processed \
  --cache_dir cache/mibig_default \
  --splits_path data/MIBIG/splits/random_seed42.tsv \
  --config projects/mibig_bgc_np/configs/default.yaml
```

Writes to `results/.../`:

- `contrastive_model_best.pt`
- `contrastive_model_last.pt`
- `contrastive_metrics.json`

### Step 3: Evaluate retrieval

Purpose:

- build unique BGC and compound embeddings for one split
- compute multi-positive retrieval metrics in both directions

Run:

```bash
python -m projects.mibig_bgc_np.scripts.eval_retrieval \
  --data_dir data/MIBIG/processed \
  --cache_dir cache/mibig_default \
  --checkpoint results/mibig_default/contrastive_model_best.pt \
  --split test \
  --config projects/mibig_bgc_np/configs/default.yaml
```

Writes:

- `results/.../retrieval_test.json`
- `results/.../embeddings_test.pt`
- `results/.../embedding_meta_test.csv`

You can also evaluate `train` or `val` with `--split`.

The output JSON format is:

```json
{
  "bgc_to_compound": {
    "mrr": 0.0,
    "recall_at_1": 0.0,
    "recall_at_5": 0.0,
    "recall_at_10": 0.0,
    "precision_at_1": 0.0,
    "precision_at_5": 0.0,
    "precision_at_10": 0.0
  },
  "compound_to_bgc": {
    "mrr": 0.0,
    "recall_at_1": 0.0,
    "recall_at_5": 0.0,
    "recall_at_10": 0.0,
    "precision_at_1": 0.0,
    "precision_at_5": 0.0,
    "precision_at_10": 0.0
  }
}
```

### Step 4: Visualize embeddings with UMAP

Purpose:

- project the joint BGC-compound embedding space into 2D
- also generate a BGC-only UMAP colored by `bgc_class`

Run:

```bash
python -m projects.mibig_bgc_np.scripts.viz_umap \
  --data_dir data/MIBIG/processed \
  --cache_dir cache/mibig_default \
  --checkpoint results/mibig_default/contrastive_model_best.pt \
  --split test \
  --config projects/mibig_bgc_np/configs/default.yaml
```

Writes under `results/.../viz/`:

- `{split}_umap.png`
- `{split}_umap_coords.csv`
- `{split}_embeddings.csv`
- `{split}_embeddings.parquet` if parquet support is installed
- `{split}_bgc_class_umap.png`
- `{split}_bgc_class_umap_coords.csv`
- `{split}_bgc_class_embeddings.csv`
- `{split}_bgc_class_embeddings.parquet` if parquet support is installed

### Step 5: Train the downstream BGC classifier

Purpose:

- freeze the learned BGC encoder from the contrastive checkpoint
- train a classifier over projected BGC embeddings
- predict `bgc_class` labels for val and test splits

Requirements:

- `mibig_pairs.tsv` must include a `bgc_class` column
- each `bgc_id` must map to exactly one `bgc_class`
- val and test labels must already appear in the training split

Run:

```bash
python -m projects.mibig_bgc_np.scripts.train_downstream \
  --data_dir data/MIBIG/processed \
  --cache_dir cache/mibig_default \
  --checkpoint results/mibig_default/contrastive_model_best.pt \
  --config projects/mibig_bgc_np/configs/default.yaml
```

If you want a different split file:

```bash
python -m projects.mibig_bgc_np.scripts.train_downstream \
  --data_dir data/MIBIG/processed \
  --cache_dir cache/mibig_default \
  --checkpoint results/mibig_default/contrastive_model_best.pt \
  --splits_path data/MIBIG/splits/random_seed42.tsv \
  --config projects/mibig_bgc_np/configs/default.yaml
```

Writes:

- `results/.../downstream_classifier.pt`
- `results/.../downstream_metrics.json`

## Config overrides

Both pipelines support dot-path config overrides through `--override`.

## Data and optional assets

This repository does not ship datasets, cached features, trained checkpoints, or the vendored `teachopencadd/` tree used for local experiments.

For the MIBiG pipeline, download the raw MIBiG release from the official download page at `https://mibig.secondarymetabolites.org/download` or the MIBiG Zenodo release for version 4.0.1 (`https://doi.org/10.5281/zenodo.14835872`). Put the files under `data/MIBIG/`, for example:

- `data/MIBIG/mibig_prot_seqs_4.0.fasta`
- `data/MIBIG/mibig_json_4.0/`

Then run `scripts/preprocess_mibig.py` to create the processed tables and split files expected by the training scripts.

As an optional side path, if you already have a pretrained KIBA checkpoint, place it anywhere under `results/` and pass it through `--checkpoint` to the KIBA evaluation, visualization, or downstream scripts. A simple convention is:

- `results/kiba_pretrained/contrastive_model_best.pt`

## Optional: KIBA workflow

This is the protein-ligand pipeline and is secondary to the BGC/MIBiG workflow above.

### What the KIBA data should look like

Your `data_dir` must contain either:

- `interS.tsv`, `lig.tsv`, `prot.tsv`
- or `kiba/resources/tables/interS.tsv`, `kiba/resources/tables/lig.tsv`, `kiba/resources/tables/prot.tsv`

Important columns:

- `interS.tsv`: `Drug_ID`, `Target_ID`, `Y`, `split`
- `lig.tsv`: ligand identifier and SMILES
- `prot.tsv`: protein identifier and amino-acid sequence

The `split` column must use `train`, `val`, and `test`.

### Step 1: Cache base features

Purpose:

- compute protein features
- compute ligand Morgan fingerprints
- save them so training does not re-encode raw data every run

Run:

```bash
python -m projects.kiba_dti.scripts.cache_features \
  --data_dir data/KIBA/datasail_kiba/kiba_R_1 \
  --outdir cache/kiba_R_1 \
  --config projects/kiba_dti/configs/default.yaml
```

Writes:

- `cache/kiba_R_1/protein_embeddings.pt`
- `cache/kiba_R_1/ligand_fingerprints.pt`
- `cache/kiba_R_1/cache_index.json`

### Step 2: Train the contrastive model

Purpose:

- train projection heads over cached protein and ligand features
- select the best checkpoint using validation retrieval

Run:

```bash
python -m projects.kiba_dti.scripts.train_contrastive \
  --data_dir data/KIBA/datasail_kiba/kiba_R_1 \
  --cache_dir cache/kiba_R_1 \
  --config projects/kiba_dti/configs/default.yaml
```

Writes to `results/.../`:

- `contrastive_model_best.pt`
- `contrastive_model_last.pt`
- `contrastive_metrics.json`

### Step 3: Evaluate retrieval

Purpose:

- build unique protein and ligand embeddings for one split
- compute retrieval metrics in both directions

Run:

```bash
python -m projects.kiba_dti.scripts.eval_retrieval \
  --data_dir data/KIBA/datasail_kiba/kiba_R_1 \
  --cache_dir cache/kiba_R_1 \
  --checkpoint results/kiba_R_1_onehot/contrastive_model_best.pt \
  --split test \
  --config projects/kiba_dti/configs/default.yaml
```

Writes:

- `results/.../retrieval_test.json`
- `results/.../embeddings_test.pt`
- `results/.../embedding_meta_test.csv`

You can switch `--split` to `val`.

### Step 4: Visualize embeddings with UMAP

Purpose:

- project the learned joint embedding space into 2D for inspection

Run:

```bash
python -m projects.kiba_dti.scripts.viz_umap \
  --data_dir data/KIBA/datasail_kiba/kiba_R_1 \
  --cache_dir cache/kiba_R_1 \
  --checkpoint results/kiba_R_1_onehot/contrastive_model_best.pt \
  --split test \
  --config projects/kiba_dti/configs/default.yaml
```

Writes under `results/.../viz/`:

- `{split}_umap.png`
- `{split}_embeddings.csv`
- `{split}_embeddings.parquet` if parquet support is installed

### Step 5: Train the downstream regressor

Purpose:

- freeze the learned CLIP encoder
- train a downstream regressor on top of the learned embeddings

Run:

```bash
python -m projects.kiba_dti.scripts.train_downstream \
  --data_dir data/KIBA/datasail_kiba/kiba_R_1 \
  --cache_dir cache/kiba_R_1 \
  --checkpoint results/kiba_R_1_onehot/contrastive_model_best.pt \
  --config projects/kiba_dti/configs/default.yaml
```

Writes:

- `results/.../downstream_regressor.pt`
- `results/.../downstream_metrics.json`

Example for KIBA:

```bash
python -m projects.kiba_dti.scripts.train_contrastive \
  --data_dir data/KIBA/datasail_kiba/kiba_R_1 \
  --cache_dir cache/kiba_R_1 \
  --config projects/kiba_dti/configs/default.yaml \
  --override train.epochs=5 model.emb_dim=128 output.dir=results/quick
```

Example for MIBiG:

```bash
python -m projects.mibig_bgc_np.scripts.train_contrastive \
  --data_dir data/MIBIG/processed \
  --cache_dir cache/mibig_default \
  --config projects/mibig_bgc_np/configs/default.yaml \
  --override train.epochs=5 output.dir=results/mibig_quick
```

## Typical outputs

### KIBA

`results/.../` typically contains:

- `contrastive_model_best.pt`
- `contrastive_model_last.pt`
- `contrastive_metrics.json`
- `retrieval_{val|test}.json`
- `embeddings_{split}.pt`
- `embedding_meta_{split}.csv`
- `viz/{split}_umap.png`
- `viz/{split}_embeddings.csv`
- `viz/{split}_embeddings.parquet`
- `downstream_regressor.pt`
- `downstream_metrics.json`

### MIBiG

`results/.../` typically contains:

- `contrastive_model_best.pt`
- `contrastive_model_last.pt`
- `contrastive_metrics.json`
- `retrieval_{train|val|test}.json`
- `embeddings_{split}.pt`
- `embedding_meta_{split}.csv`
- `viz/{split}_umap.png`
- `viz/{split}_umap_coords.csv`
- `viz/{split}_embeddings.csv`
- `viz/{split}_embeddings.parquet`
- `viz/{split}_bgc_class_umap.png`
- `viz/{split}_bgc_class_umap_coords.csv`
- `viz/{split}_bgc_class_embeddings.csv`
- `viz/{split}_bgc_class_embeddings.parquet`
- `downstream_classifier.pt`
- `downstream_metrics.json`
