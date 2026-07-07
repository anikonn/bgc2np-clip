# BGC2NP-CLIP

CLIP-style retrieval pipeline for matching MIBiG biosynthetic gene clusters
(BGCs) with natural products.

## What This Repo Contains

```text
src/clip_core/                  Shared config, logging, cache, loss, and retrieval helpers
src/mibig_clip/                 MIBiG retrieval and visualization helpers
projects/mibig_bgc_np/          MIBiG models, featurization, training, evaluation, and configs
scripts/preprocess_mibig.py     Convert raw MIBiG files into processed training tables
scripts/make_mibig_splits.py    Generate train/val/test splits or CV folds
scripts/train_downstream.py     Wrapper for the downstream task entrypoint
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

Optional dependency for parquet UMAP exports:

```bash
pip install pyarrow
```

## Data

Datasets, cached features, and trained checkpoints are not included in this
repository.

Download the raw MIBiG release from the official MIBiG download page
(`https://mibig.secondarymetabolites.org/download`) or the MIBiG Zenodo release
(`https://doi.org/10.5281/zenodo.14835872`), then place the files under
`data/MIBIG/`:

```text
data/MIBIG/mibig_prot_seqs_4.0.fasta
data/MIBIG/mibig_json_4.0/
```

For NPAtlas-backed retrieval and downstream compound tasks, also download:

```text
https://www.npatlas.org/static/downloads/NPAtlas_download.tsv
```

Place it under `data/`, for example:

```text
data/NPAtlas_download.tsv
```

Then run the preprocessing step below to create the processed tables expected by
the training scripts.

## Pipeline Overview

The standard workflow is:

1. Preprocess raw MIBiG files.
2. Create split files.
3. Cache BGC and compound features.
4. Train the contrastive model.
5. Evaluate retrieval.
6. Visualize embeddings.
7. Train downstream prediction tasks.

Most pipeline steps should be run as Python modules:

```bash
python -m projects.mibig_bgc_np.scripts.<script_name>
```

Top-level helper scripts are available for preprocessing, split generation, and
the downstream wrapper.

## Expected Data

Training and evaluation expect these processed files:

```text
data/MIBIG/processed/mibig_pairs.tsv
data/MIBIG/processed/bgc_proteins.jsonl
data/MIBIG/splits/<split_file>.tsv
```

You can use another split TSV. BGC-level split files are keyed by `bgc_id`;
NP-level split files are keyed by both `bgc_id` and `compound_id`.

`mibig_pairs.tsv` should contain:

- `bgc_id`
- a compound identifier column, such as `compound_id`, `canonical_smiles`, or `smiles`
- `bgc_classes` for downstream multilabel BGC class prediction
- either a `split` column or a separate split TSV

Optional columns:

- `n_bgc_classes`, used for preprocessing summaries and single-class filtering metadata
- `bgc_class`, kept as a compatibility alias for `bgc_classes`

`bgc_proteins.jsonl` should map each `bgc_id` to `protein_ids` and
`protein_seqs`.

## Step 1: Preprocess Raw MIBiG

Use this when starting from raw MIBiG FASTA and JSON files:

```bash
python scripts/preprocess_mibig.py \
  --fasta_path data/MIBIG/mibig_prot_seqs_4.0.fasta \
  --json_dir data/MIBIG/mibig_json_4.0 \
  --out_dir data/MIBIG/processed
```

Inputs:

```text
data/MIBIG/mibig_prot_seqs_4.0.fasta
data/MIBIG/mibig_json_4.0/
```

Outputs:

```text
data/MIBIG/processed/mibig_pairs.tsv
data/MIBIG/processed/bgc_proteins.jsonl
data/MIBIG/processed/mibig_summary.json
```

Notes:

- BGCs with more than 3000 proteins are dropped during preprocessing.
- `bgc_classes` preserves all annotated biosynthetic classes as a semicolon-separated multilabel field.
- `bgc_class` mirrors the same semicolon-separated value for compatibility.

## Step 2: Create Splits

Choose a split type with `--split_type`:

- `bgc`: BGC-based split. Each BGC stays in one split or fold, but the same natural product can appear in multiple splits through different BGCs.
- `combined`: BGC-NP connected-component split. BGCs connected by shared natural products stay together, so neither BGCs nor natural products cross splits or folds.
- `np`: NP-based split. Each natural product stays in one split or fold, but a multi-product BGC can appear in multiple splits through different products.
- `strict`: BiG-SCAPE family + NP Butina connected-component split. Use this for the strongest leakage-aware CV setting.

The default is `combined`.

Create a random train/val/test split:

```bash
python scripts/make_mibig_splits.py \
  --pairs_path data/MIBIG/processed/mibig_pairs.tsv \
  --splits_dir data/MIBIG/splits \
  --seed 42 \
  --split_mode random \
  --split_type combined
```

Output:

```text
data/MIBIG/splits/combined_random_seed42.tsv
```

For `--split_type bgc` or `--split_type np`, the random split filenames are
`bgc_random_seed42.tsv` and `np_random_seed42.tsv`.

Create cross-validation fold assignments:

```bash
python scripts/make_mibig_splits.py \
  --pairs_path data/MIBIG/processed/mibig_pairs.tsv \
  --splits_dir data/MIBIG/splits \
  --seed 42 \
  --split_mode cv \
  --split_type combined \
  --n_folds 10
```

Output:

```text
data/MIBIG/splits/combined_cv_seed42_n10.tsv
```

For `--split_type bgc` or `--split_type np`, the CV filenames are
`bgc_cv_seed42_n10.tsv` and `np_cv_seed42_n10.tsv`.

For CV files, pass `--cv_fold K` to training and evaluation scripts. The selected
fold is used as test data and all other folds are used for training.

### BCCoE-Style CV Splits

To compare against the BCCoE 10-fold cross-validation protocol described by
the Scientific Reports paper published on 2026-05-09
(`https://www.nature.com/articles/s41598-026-49955-5`), create two seed-0 split
files:

- `bccoe_bgc`: BGC-to-chemical retrieval setting. BGCs are randomly assigned to
  10 folds, and all positive pairs for a BGC stay in one fold.
- `bccoe_np`: chemical-to-BGC retrieval setting. Natural products are randomly
  assigned to 10 folds, and all positive pairs for a natural product stay in
  one fold.

These are protocol-compatible recreations from the local `mibig_pairs.tsv`
table, not author-provided fold files.

If you want the closer BCCoE-style preprocessing, first build the PFAM/SMILES
filtered dataset and OHE+Morgan cache from the files in `data/MIBIG/bccoe`:

```bash
python scripts/make_bccoe_processed.py \
  --pairs_path data/MIBIG/processed/mibig_pairs.tsv \
  --proteins_path data/MIBIG/processed/bgc_proteins.jsonl \
  --pfam_path data/MIBIG/bccoe/cand_BGCs_pfams.csv \
  --out_dir cache/bccoe

python -m projects.mibig_bgc_np.scripts.cache_features \
  --data_dir cache/bccoe \
  --out_dir cache/bccoe \
  --config projects/mibig_bgc_np/configs/ohe.yaml
```

This keeps paired BGCs with non-empty PFAM annotations, canonicalizes SMILES
with RDKit, and deduplicates `bgc_id`/canonical-SMILES pairs. With the current
local MIBiG table this gives 2105 BGCs, 3463 compounds, and 3806 pairs. The
provided PFAM file contains the paper's 2625 PFAM-covered BGCs, but only 2105 of
those have paired SMILES in the local `mibig_pairs.tsv`.

```bash
python scripts/make_mibig_splits.py \
  --pairs_path data/MIBIG/processed/mibig_pairs.tsv \
  --splits_dir data/MIBIG/splits \
  --split_mode cv \
  --split_type bgc \
  --seed 0 \
  --n_folds 10 \
  --output_prefix bccoe_bgc

python scripts/make_mibig_splits.py \
  --pairs_path data/MIBIG/processed/mibig_pairs.tsv \
  --splits_dir data/MIBIG/splits \
  --split_mode cv \
  --split_type np \
  --seed 0 \
  --n_folds 10 \
  --output_prefix bccoe_np
```

Outputs:

```text
data/MIBIG/splits/bccoe_bgc_cv_seed0_n10.tsv
data/MIBIG/splits/bccoe_np_cv_seed0_n10.tsv
```

For the filtered `cache/bccoe` dataset, generate matching split files in the
same directory:

```bash
python scripts/make_mibig_splits.py \
  --pairs_path cache/bccoe/mibig_pairs.tsv \
  --splits_dir cache/bccoe \
  --split_mode cv \
  --split_type bgc \
  --seed 0 \
  --n_folds 10 \
  --output_prefix bccoe_bgc

python scripts/make_mibig_splits.py \
  --pairs_path cache/bccoe/mibig_pairs.tsv \
  --splits_dir cache/bccoe \
  --split_mode cv \
  --split_type np \
  --seed 0 \
  --n_folds 10 \
  --output_prefix bccoe_np
```

### Leave-One-Class-Out Splits

Use leave-one-BGC-product-class-out splits to test extrapolation to novel
product classes. This recreates the paper's two class-holdout settings:

- `exp3_bgc`: BGC-to-chemical retrieval. For each target class, BGCs annotated
  with that class and all their positive pairs are test data; all other BGCs are
  training data.
- `exp4_np`: chemical-to-BGC retrieval. For each target class, compounds linked
  to BGCs annotated with that class and all their positive pairs are test data;
  all other compounds are training data.

BGCs and compounds can be multi-class, so class-specific test percentages can
sum to more than 100%.

```bash
python scripts/make_leave_one_class_out_splits.py \
  --pairs_path cache/bccoe/mibig_pairs.tsv \
  --out_dir cache/bccoe/leave_one_class_out \
  --mode both \
  --prefix bccoe_loco
```

Outputs:

```text
cache/bccoe/leave_one_class_out/exp3_bgc/bccoe_loco_exp3_bgc_<class>.tsv
cache/bccoe/leave_one_class_out/exp4_np/bccoe_loco_exp4_np_<class>.tsv
cache/bccoe/leave_one_class_out/bccoe_loco_leave_one_class_out_summary.tsv
```

### NP Butina Clusters Only

Use this preparatory step when you only want chemical NP/product clusters and do
not yet want to build the final strict leakage-aware split. The final strict
split should be created later by combining this NP Butina table with externally
provided BGC cluster assignments.

The script loads the final MiBIG BGC-NP paired table, canonicalizes product
SMILES with RDKit, drops invalid SMILES into a separate report, computes Morgan
fingerprints with radius 2 and 2048 bits, and clusters unique valid compounds
with RDKit Butina using distance threshold `0.3`. Because Butina uses distance,
this corresponds to fixed Tanimoto similarity `>= 0.7`.

Run from an environment with RDKit installed:

```bash
python scripts/make_mibig_np_butina_clusters.py \
  --pairs_path data/MIBIG/processed/mibig_pairs.tsv \
  --output_path data/MIBIG/processed/mibig_np_butina_clusters_tanimoto0.7.tsv
```

If the environment is not already activated, use the same command through conda:

```bash
conda run -n combi python scripts/make_mibig_np_butina_clusters.py \
  --pairs_path data/MIBIG/processed/mibig_pairs.tsv \
  --output_path data/MIBIG/processed/mibig_np_butina_clusters_tanimoto0.7.tsv
```

Main output:

```text
data/MIBIG/processed/mibig_np_butina_clusters_tanimoto0.7.tsv
```

The output table contains:

```text
product_id, compound_name, compound_key, compound_key_source,
smiles, canonical_smiles, np_butina_cluster
```

For the current `mibig_pairs.tsv`, there is no stable explicit product ID
column, so `compound_key_source` is expected to be `canonical_smiles`. If a
future pair table contains `product_id`, `compound_id`, `np_id`, `npatlas_id`,
or `database_id`, the script will prefer that as the product key.

Sidecar outputs:

```text
data/MIBIG/processed/mibig_np_butina_clusters_tanimoto0.7_invalid_smiles.tsv
data/MIBIG/processed/mibig_np_butina_clusters_tanimoto0.7_cluster_sizes.tsv
data/MIBIG/processed/mibig_np_butina_clusters_tanimoto0.7_report.json
```

The report JSON includes:

```text
n_unique_compounds_before_filtering
n_valid_unique_compounds
n_invalid_smiles
n_butina_clusters
n_singleton_clusters
largest_cluster_size
median_cluster_size
cluster_size_distribution
```

This step does not run BiG-SCAPE and does not create train/validation/test
splits.

### Strict Leakage-Aware Splits

Use this when you want the stricter split based on both BiG-SCAPE BGC families
and NP-side Butina clusters. The script uses unique product SMILES from
`data/MIBIG/processed/mibig_pairs.tsv`, canonicalizes them with RDKit, clusters
them with Morgan radius 2 / 2048-bit fingerprints and Butina distance threshold
0.3, then builds leakage groups from a bipartite graph:

```text
BiG-SCAPE family node -- observed BGC-NP pair -- NP Butina cluster node
```

Connected components of this graph become `leakage_group`. The script checks
that no BiG-SCAPE family and no NP Butina cluster appears in more than one
random split or CV fold.

If BiG-SCAPE family assignments and NP Butina clusters have already been
prepared, create both strict train/validation/test and strict 10-fold CV splits
directly from those tables:

```bash
python -m scripts.make_strict_modal_leakage_splits \
  --pairs_path data/MIBIG/processed/mibig_pairs.tsv \
  --bigscape_path data/MIBIG/processed/bigscape_clustering.tsv \
  --butina_path data/MIBIG/processed/mibig_np_butina_clusters_tanimoto0.7.tsv \
  --out_dir data/MIBIG/processed/strict_splits \
  --seed 42 \
  --n_folds 10
```

This assigns whole connected-component leakage groups to splits or CV folds and
writes:

```text
data/MIBIG/processed/strict_splits/mibig_pairs_with_leakage_groups.tsv
data/MIBIG/processed/strict_splits/mibig_pairs_strict_train_val_test.tsv
data/MIBIG/processed/strict_splits/mibig_pairs_strict_cv10.tsv
data/MIBIG/processed/strict_splits/mibig_pairs_strict_cv10_long.tsv
data/MIBIG/processed/strict_splits/strict_train_val_test_summary.tsv
data/MIBIG/processed/strict_splits/strict_cv10_summary.tsv
data/MIBIG/processed/strict_splits/largest_leakage_groups.tsv
data/MIBIG/processed/strict_splits/missing_cluster_report.tsv
data/MIBIG/processed/strict_splits/strict_split_sanity_checks.txt
```

For BiG-SCAPE family assignment, the script first reads
`record_annotations.tsv` as the complete set of records processed by BiG-SCAPE.
It then searches every `*_clustering_c0.3.tsv` under the BiG-SCAPE output
directory and merges explicit family assignments onto the processed records.
Processed records absent from all clustering TSVs are assigned singleton
families as `SINGLETON_<bgc_id>`. Only paired BGCs absent from
`record_annotations.tsv` are reported as truly unprocessed by BiG-SCAPE.

Create the strict 10-fold CV split:

```bash
python -m scripts.make_strict_leakage_splits \
  --pairs_path data/MIBIG/processed/mibig_pairs.tsv \
  --bigscape_output_dir data/MIBIG/mibig_bigscape_clustered/output_files/2026-06-24_18-27-02_c0.3 \
  --splits_dir data/MIBIG/splits \
  --split_mode cv \
  --n_folds 10 \
  --seed 42
```

Run this from an environment with RDKit installed. `conda run -n combi` is not
obligatory; it is only a convenience if the environment is not already
activated. In an activated environment, plain `python -m ...` is preferred.

CV outputs:

```text
data/MIBIG/splits/strict_bigscape_butina_np_butina_clusters.tsv
data/MIBIG/splits/strict_bigscape_butina_invalid_smiles.tsv
data/MIBIG/splits/strict_bigscape_butina_cv_seed42_n10.tsv
data/MIBIG/splits/strict_bigscape_butina_cv_seed42_n10_report.json
```

The CV split TSV contains:

```text
bgc_id, compound_id, smiles, canonical_smiles, fold_id,
leakage_group, bigscape_family, bigscape_family_source, np_butina_cluster
```

Create a strict random train/validation/test split instead:

```bash
python -m scripts.make_strict_leakage_splits \
  --pairs_path data/MIBIG/processed/mibig_pairs.tsv \
  --bigscape_output_dir data/MIBIG/mibig_bigscape_clustered/output_files/2026-06-24_18-27-02_c0.3 \
  --splits_dir data/MIBIG/splits \
  --seed 42
```

Outputs:

```text
data/MIBIG/splits/strict_bigscape_butina_np_butina_clusters.tsv
data/MIBIG/splits/strict_bigscape_butina_invalid_smiles.tsv
data/MIBIG/splits/strict_bigscape_butina_random_seed42.tsv
data/MIBIG/splits/strict_bigscape_butina_random_seed42_report.json
```

The random split TSV contains:

```text
bgc_id, compound_id, smiles, canonical_smiles, split,
leakage_group, bigscape_family, bigscape_family_source, np_butina_cluster
```

The report JSON includes:

```text
bigscape_family_assignment.n_paired_bgcs
bigscape_family_assignment.n_matched_to_bigscape_processed_records
bigscape_family_assignment.n_with_explicit_bigscape_family
bigscape_family_assignment.n_assigned_singleton_family
bigscape_family_assignment.n_still_missing_completely
```

By default, paired BGCs that are absent from `record_annotations.tsv` are kept as
`UNPROCESSED_BIGSCAPE_<bgc_id>` singleton families and reported. Use
`--fail_on_missing_bigscape` if you prefer the script to stop for those truly
unprocessed BGCs.

## Step 3: Cache Features

```bash
python -m projects.mibig_bgc_np.scripts.cache_features \
  --data_dir data/MIBIG/processed \
  --out_dir cache/ohe \
  --config projects/mibig_bgc_np/configs/ohe.yaml
```

For the BGC-MAP retrieval benchmark, build a cache that also includes MAP
negative candidate products. Pass the raw FASTA too, so BGCs that are present in
the MAP split but absent from `bgc_proteins.jsonl` can still be cached:

```bash
python -m projects.mibig_bgc_np.scripts.cache_features \
  --data_dir data/MIBIG/processed \
  --out_dir cache/mibig_map \
  --config projects/mibig_bgc_np/configs/default.yaml \
  --map_metadata_path data/MIBIG/splits/MAP_metadata_fold.csv \
  --fasta_path data/MIBIG/mibig_prot_seqs_4.0.fasta
```

The MAP cache must contain features for every BGC/product row that will be
evaluated. The runner intentionally raises an error if BGCs or products are
missing, because dropping rows would change the benchmark counts.

For the full BGC-MAC classification setting, build a cache that also includes
all BGCs from `bgcmac_fold.csv`. These extra BGCs can lack product SMILES; they
are not used for CLIP pair training, but they can be embedded by the BGC encoder
for downstream BGC class prediction:

```bash
python -m projects.mibig_bgc_np.scripts.cache_features \
  --data_dir data/MIBIG/processed \
  --out_dir cache/ohe \
  --config projects/mibig_bgc_np/configs/ohe.yaml \
  --bgcmac_splits_path data/MIBIG/splits/bgcmac_fold.csv \
  --fasta_path data/MIBIG/mibig_prot_seqs_4.0.fasta
```

Outputs:

```text
cache/ohe/bgc_features.pt
cache/ohe/compound_features.pt
cache/ohe/cache_index.json
```

Protein encoder options are configured with `featurization.bgc_encoder` in
`projects/mibig_bgc_np/configs/ohe.yaml`:

- `ohe`: one-hot encode residues, then mean-pool protein embeddings within each BGC
- `esm2_mean`: encode each protein with ESM2 and mean-pool residue embeddings
- `esm2_cls`: encode each protein with ESM2 and use the BOS/CLS token embedding

The `ohe.yaml` config uses the `ohe` encoder.

## Step 4: Train Contrastive Model

Default training command:

```bash
python -m projects.mibig_bgc_np.scripts.train_contrastive \
  --data_dir data/MIBIG/processed \
  --cache_dir cache/ohe \
  --config projects/mibig_bgc_np/configs/ohe.yaml
```

Use an explicit split file:

```bash
python -m projects.mibig_bgc_np.scripts.train_contrastive \
  --data_dir data/MIBIG/processed \
  --cache_dir cache/ohe \
  --splits_path data/MIBIG/splits/combined_random_seed42.tsv \
  --config projects/mibig_bgc_np/configs/ohe.yaml
```

Use a CV fold:

```bash
python -m projects.mibig_bgc_np.scripts.train_contrastive \
  --data_dir data/MIBIG/processed \
  --cache_dir cache/ohe \
  --splits_path data/MIBIG/splits/combined_cv_seed42_n10.tsv \
  --cv_fold 1 \
  --config projects/mibig_bgc_np/configs/ohe.yaml
```

Outputs are written under `results/.../`:

```text
contrastive_model_best.pt
contrastive_model_last.pt
contrastive_metrics.json
```

## Step 5: Evaluate Retrieval

Evaluate against the in-dataset candidate set:

```bash
python -m projects.mibig_bgc_np.scripts.eval_retrieval \
  --data_dir data/MIBIG/processed \
  --cache_dir cache/ohe \
  --checkpoint results/mibig_default/contrastive_model_best.pt \
  --split test \
  --config projects/mibig_bgc_np/configs/ohe.yaml
```

Evaluate with an optional NPAtlas candidate library:

```bash
python -m projects.mibig_bgc_np.scripts.eval_retrieval \
  --data_dir data/MIBIG/processed \
  --cache_dir cache/ohe \
  --checkpoint results/mibig_default/contrastive_model_best.pt \
  --split test \
  --npatlas_path data/NPAtlas_download_2024_09.tsv \
  --npatlas_n 10000 \
  --seed 42 \
  --config projects/mibig_bgc_np/configs/ohe.yaml
```

Outputs:

```text
results/.../retrieval/in_dataset_retrieval.json
results/.../retrieval/npatlas_retrieval_n{n}.json
results/.../retrieval/embeddings_test.pt
results/.../retrieval/embedding_meta_test.csv
```

Retrieval metrics include MRR, median rank, recall@k, precision@k, eligible BGC
counts, skipped-match counts, and match counts by InChIKey or SMILES.

## Retrieval Baselines

The CV, BGC-MAC, and BGC-MAP runners also write retrieval baselines for the
same train/validation/test splits used by the CLIP model. They are enabled by
default and can be disabled with `--no_retrieval_baselines`.

Implemented baselines:

- `random`: randomly ranks candidates with a fixed seed and reports mean/std
  across `--baseline_random_trials` trials.
- `frozen_encoder_similarity`: uses the cached frozen BGC and NP input features
  directly, without contrastive training. If feature dimensions differ, both
  modalities are mapped into the configured embedding dimension with fixed
  non-learned Gaussian random projections, then candidates are ranked by cosine
  similarity.
- `linear_projection`: freezes the cached BGC and NP input features, replaces
  nonlinear CLIP heads with linear projections, trains with the same symmetric
  multi-positive InfoNCE objective, and evaluates by cosine similarity.
- `knn_transfer`: transfers scores from nearest training pairs. BGC-to-NP uses
  frozen BGC feature cosine similarity; NP-to-BGC uses Tanimoto similarity over
  cached NP fingerprint features. The default `k` values are 1, 5, and 10.

All baselines report the same bidirectional retrieval metric structure as the
CLIP model:

```text
bgc_to_compound: MRR, Recall@1, Recall@5, Recall@10, Recall@20, Recall@50, Recall@100, Recall@200, Recall@500
compound_to_bgc: MRR, Recall@1, Recall@5, Recall@10, Recall@20, Recall@50, Recall@100, Recall@200, Recall@500
```

The JSON also includes precision@k because the main retrieval evaluator already
tracks it. Output files are written per fold/member under:

```text
retrieval_baselines/retrieval_baselines_test.json
```

The runners also collect visible baseline summaries and plots at the run root:

```text
results/<run_name>/baselines/baseline_artifacts.json
results/<run_name>/baselines/retrieval/retrieval_baselines_summary.csv
results/<run_name>/baselines/retrieval/retrieval_baselines_mrr.png
results/<run_name>/baselines/retrieval/retrieval_baselines_recall_at_1.png
results/<run_name>/baselines/retrieval/retrieval_baselines_recall_at_5.png
results/<run_name>/baselines/retrieval/retrieval_baselines_recall_at_10.png
results/<run_name>/baselines/retrieval/retrieval_baselines_recall_at_20.png
results/<run_name>/baselines/retrieval/retrieval_baselines_recall_at_50.png
results/<run_name>/baselines/retrieval/retrieval_baselines_recall_at_100.png
results/<run_name>/baselines/retrieval/retrieval_baselines_recall_at_200.png
results/<run_name>/baselines/retrieval/retrieval_baselines_recall_at_500.png
results/<run_name>/baselines/classification/classification_baselines_summary.csv
results/<run_name>/baselines/classification/classification_baselines_overall.png
results/<run_name>/baselines/classification/classification_baselines_per_class_auroc.png
results/<run_name>/baselines/classification/classification_baselines_per_class_f1.png
```

For an existing run, regenerate only this visible baseline folder without
retraining:

```bash
python -m projects.mibig_bgc_np.scripts.package_baseline_artifacts \
  --run_root results/combined_cv10
```

Useful baseline options:

```text
--baseline_random_trials 10
--baseline_k_values 1 5 10
--no_retrieval_baselines
```

## Step 6: Visualize Embeddings

```bash
python -m projects.mibig_bgc_np.scripts.viz_umap \
  --data_dir data/MIBIG/processed \
  --cache_dir cache/ohe \
  --checkpoint results/mibig_default/contrastive_model_best.pt \
  --split test \
  --config projects/mibig_bgc_np/configs/ohe.yaml
```

Outputs are written under `results/.../viz/`:

```text
{split}_umap.png
{split}_umap_coords.csv
{split}_embeddings.csv
{split}_embeddings.parquet
{split}_bgc_class_umap.png
{split}_bgc_class_umap_coords.csv
{split}_bgc_class_embeddings.csv
{split}_bgc_class_embeddings.parquet
```

Parquet files are only written when parquet support is installed.

## Step 7: Train Downstream Tasks

```bash
python -m projects.mibig_bgc_np.scripts.train_downstream \
  --data_dir data/MIBIG/processed \
  --cache_dir cache/ohe \
  --checkpoint results/mibig_default/contrastive_model_best.pt \
  --config projects/mibig_bgc_np/configs/ohe.yaml
```

Equivalent wrapper:

```bash
python scripts/train_downstream.py \
  --data_dir data/MIBIG/processed \
  --cache_dir cache/ohe \
  --checkpoint results/mibig_default/contrastive_model_best.pt \
  --config projects/mibig_bgc_np/configs/ohe.yaml
```

By default, all downstream tasks run:

- `bgc_class`
- `compound_mw`
- `origin_type`

Use repeated `--task` flags to run a subset:

```bash
python -m projects.mibig_bgc_np.scripts.train_downstream \
  --data_dir data/MIBIG/processed \
  --cache_dir cache/ohe \
  --checkpoint results/mibig_default/contrastive_model_best.pt \
  --task bgc_class \
  --task origin_type \
  --config projects/mibig_bgc_np/configs/ohe.yaml
```

Useful optional flags:

- `--splits_path data/MIBIG/splits/combined_random_seed42.tsv`
- `--cv_fold 1`
- `--mibig_pairs_path data/MIBIG/processed/mibig_pairs.tsv`
- `--npatlas_path data/NPAtlas_download_2024_09.tsv`
- `--save_cm_png`

Common downstream outputs:

```text
downstream_metrics.json
downstream_classifier.pt
matched_compounds.tsv
```

Task-specific outputs:

- `bgc_class`: multilabel metrics, per-class metrics, ROC curve coordinates, and optional confusion-matrix and ROC PNGs
- `origin_type`: `downstream_origin_type_metrics.json`, `downstream_origin_type_dataset_stats.json`, and optional confusion-matrix PNGs
- `compound_mw`: `downstream_compound_mw_metrics.json`, `downstream_mw_hist.png`, `downstream_mw_by_bgc_class.png`, and `downstream_mw_by_origin_type.png`

## Cross-Validation

Run grouped cross-validation without recaching features:

```bash
python -m projects.mibig_bgc_np.scripts.run_cv10 \
  --data_dir data/MIBIG/processed \
  --cache_dir cache/ohe \
  --config projects/mibig_bgc_np/configs/ohe.yaml \
  --seed 42 \
  --n_folds 10
```

CV saves per-fold downstream confusion matrices, ROC plots, and aggregate CV
summary confusion matrices by default. Use `--no_save_cm_png` only if you want
to skip those PNG outputs.

The CV runner also writes a cached-BGC-feature MLP classifier baseline by
default. This is the OHE+MLP baseline when `cache_dir` was built with the OHE
BGC encoder, and the same MLP baseline over ESM features when the cache was
built with ESM. It does not use CLIP embeddings.

```text
results/{split_type}_cv{n_folds}/fold_{k}/raw_bgc_classifier_baseline/raw_bgc_metrics.json
results/{split_type}_cv{n_folds}/raw_bgc_classifier_baseline_summary.json
```

Use `--no_raw_bgc_classifier_baseline` to disable it.

By default, `run_cv10` uses `--split_type combined` and looks for:

```text
data/MIBIG/splits/combined_cv_seed42_n10.tsv
```

To use a different split type, pass `--split_type`:

```bash
python -m projects.mibig_bgc_np.scripts.run_cv10 \
  --data_dir data/MIBIG/processed \
  --cache_dir cache/ohe \
  --config projects/mibig_bgc_np/configs/ohe.yaml \
  --seed 42 \
  --n_folds 10 \
  --split_type bgc
```

CV split filenames follow the selected split type:

```text
data/MIBIG/splits/bgc_cv_seed42_n10.tsv
data/MIBIG/splits/combined_cv_seed42_n10.tsv
data/MIBIG/splits/np_cv_seed42_n10.tsv
data/MIBIG/splits/strict_bigscape_butina_cv_seed42_n10.tsv
```

Run strict leakage-free CV10 with:

```bash
python -m projects.mibig_bgc_np.scripts.run_cv10 \
  --data_dir data/MIBIG/processed \
  --cache_dir cache/ohe \
  --config projects/mibig_bgc_np/configs/ohe.yaml \
  --seed 42 \
  --n_folds 10 \
  --split_type strict
```

If you are using the strict split table generated under `processed/strict_splits`,
pass it explicitly:

```bash
python -m projects.mibig_bgc_np.scripts.run_cv10 \
  --data_dir data/MIBIG/processed \
  --cache_dir cache/ohe \
  --config projects/mibig_bgc_np/configs/ohe.yaml \
  --seed 42 \
  --n_folds 10 \
  --split_type strict \
  --splits_path data/MIBIG/processed/strict_splits/mibig_pairs_strict_cv10.tsv \
  --run_name strict_modal_cv10 \
  --baseline_k_values 1 5 10 20 50 100 200 500
```

After CV finishes, create retrieval plots for both retrieval directions:

```bash
python -m projects.mibig_bgc_np.scripts.plot_retrieval_summary \
  --summary results/strict_modal_cv10/summary.json \
  --outdir results/strict_modal_cv10/retrieval_plots \
  --prefix strict_modal
```

This writes Top-K Recall plots, an MRR bar plot for the baselines and model,
and aggregate retrieval BGC-class ROC/confusion plots when
`retrieval_class_test.classes` is present in the summary.

You can also bypass the naming convention with an explicit split file:

```bash
python -m projects.mibig_bgc_np.scripts.run_cv10 \
  --data_dir data/MIBIG/processed \
  --cache_dir cache/ohe \
  --config projects/mibig_bgc_np/configs/ohe.yaml \
  --splits_path data/MIBIG/splits/np_cv_seed42_n10.tsv \
  --n_folds 10
```

Run the BCCoE-style seed-0 BGC and NP/chemical CV comparisons with explicit
split files and named result roots:

```bash
python -m projects.mibig_bgc_np.scripts.run_cv10 \
  --data_dir cache/bccoe \
  --cache_dir cache/bccoe \
  --config projects/mibig_bgc_np/configs/ohe.yaml \
  --seed 0 \
  --n_folds 10 \
  --split_type bgc \
  --splits_path cache/bccoe/bccoe_bgc_cv_seed0_n10.tsv \
  --run_name bccoe_bgc_cv10 \
  --baseline_k_values 1 5 10 20 50 100 200 500

python -m projects.mibig_bgc_np.scripts.run_cv10 \
  --data_dir cache/bccoe \
  --cache_dir cache/bccoe \
  --config projects/mibig_bgc_np/configs/ohe.yaml \
  --seed 0 \
  --n_folds 10 \
  --split_type np \
  --splits_path cache/bccoe/bccoe_np_cv_seed0_n10.tsv \
  --run_name bccoe_np_cv10 \
  --baseline_k_values 1 5 10 20 50 100 200 500
```

After the runs finish, create BCCoE-style Top-K Recall bar plots for
`K = 5, 10, 20, 50, 100, 200, 500`. The plot includes the trained model plus
`random`, `frozen_encoder_similarity`, and `KNN-5` when those baselines are
present in the run summary. No other baseline variants are included in this
compact BCCoE figure.

```bash
python -m projects.mibig_bgc_np.scripts.plot_bccoe_retrieval \
  --summary results/bccoe_bgc_cv10/summary.json \
  --outdir results/bccoe_bgc_cv10/bccoe_retrieval_plots \
  --prefix bccoe_bgc \
  --model_label Combi \
  --directions bgc_to_compound

python -m projects.mibig_bgc_np.scripts.plot_bccoe_retrieval \
  --summary results/bccoe_np_cv10/summary.json \
  --outdir results/bccoe_np_cv10/bccoe_retrieval_plots \
  --prefix bccoe_np \
  --model_label Combi \
  --directions compound_to_bgc
```

Plot outputs:

```text
results/bccoe_bgc_cv10/bccoe_retrieval_plots/bccoe_bgc_bgc_to_np_topk_recall.png
results/bccoe_np_cv10/bccoe_retrieval_plots/bccoe_np_np_to_bgc_topk_recall.png
results/bccoe_bgc_cv10/bccoe_retrieval_plots/bccoe_bgc_long.csv
results/bccoe_bgc_cv10/bccoe_retrieval_plots/bccoe_bgc_summary.csv
```

Create the BCCoE paper comparison table for top-10 and top-100 recall across
the four BCCoE-style experiments:

```bash
python -m projects.mibig_bgc_np.scripts.make_bccoe_comparison_table \
  --outdir results/bccoe_comparison_table
```

The table uses our `Random`, `Frozen`, and `KNN-5` baselines, copies the paper's
BCCoE recall numbers, and adds `BGC2NP-CLIP` as the final row. Lift is not
included. Our values use the same aggregation as the BCCoE recall plots:
`recall_at_K` is averaged over CV folds or leave-one-class-out holdouts and then
reported as a percentage.

```text
results/bccoe_comparison_table/bccoe_recall_compact.png
results/bccoe_comparison_table/bccoe_recall_compact.md
results/bccoe_comparison_table/bccoe_recall_by_method.csv
results/bccoe_comparison_table/bccoe_recall_long.csv
```

Run leave-one-class-out retrieval experiments with the same Top-K baseline set:

```bash
python -m projects.mibig_bgc_np.scripts.run_leave_one_class_out \
  --data_dir cache/bccoe \
  --cache_dir cache/bccoe \
  --splits_dir cache/bccoe/leave_one_class_out/exp3_bgc \
  --split_glob "bccoe_loco_exp3_bgc_*.tsv" \
  --config projects/mibig_bgc_np/configs/ohe.yaml \
  --seed 0 \
  --run_name bccoe_loco_exp3_bgc \
  --baseline_k_values 1 5 10 20 50 100 200 500

python -m projects.mibig_bgc_np.scripts.run_leave_one_class_out \
  --data_dir cache/bccoe \
  --cache_dir cache/bccoe \
  --splits_dir cache/bccoe/leave_one_class_out/exp4_np \
  --split_glob "bccoe_loco_exp4_np_*.tsv" \
  --config projects/mibig_bgc_np/configs/ohe.yaml \
  --seed 0 \
  --run_name bccoe_loco_exp4_np \
  --baseline_k_values 1 5 10 20 50 100 200 500
```

Equivalent Condor wrappers:

```bash
condor/run_loco_bgc.sh
condor/run_loco_np.sh
```

After the LOCO runs finish, reuse the Top-K Recall plotter:

```bash
python -m projects.mibig_bgc_np.scripts.plot_bccoe_retrieval \
  --summary results/bccoe_loco_exp3_bgc/summary.json \
  --outdir results/bccoe_loco_exp3_bgc/loco_retrieval_plots \
  --prefix bccoe_loco_exp3_bgc \
  --model_label Combi \
  --directions bgc_to_compound

python -m projects.mibig_bgc_np.scripts.plot_bccoe_retrieval \
  --summary results/bccoe_loco_exp4_np/summary.json \
  --outdir results/bccoe_loco_exp4_np/loco_retrieval_plots \
  --prefix bccoe_loco_exp4_np \
  --model_label Combi \
  --directions compound_to_bgc
```

For each fold `k`:

- train: `fold_id != k`
- test: `fold_id == k`

Per-fold outputs are written under:

```text
results/combined_cv10/fold_1/
results/combined_cv10/fold_2/
...
results/combined_cv10/fold_10/
```

The aggregated summary is written to:

```text
results/combined_cv10/summary.json
```

Aggregate CV summary plots are written under:

```text
results/combined_cv10/summary_confusion_matrices/
```

The output root includes the split type:

```text
results/bgc_cv10/
results/combined_cv10/
results/np_cv10/
```

If you use `--n_folds N`, the output root becomes `results/{split_type}_cvN/`.

## BGC-MAC Classification Benchmark

BGC-MAC is the BGC class prediction benchmark. It uses
`data/MIBIG/splits/bgcmac_fold.csv`, where fold 10 is the fixed test fold and
folds 1-9 are rotated as validation folds to train nine ensemble members.

### BGC-MAC Cache

Build the BGC-MAC cache with the split file and raw FASTA if you want the
full-BGC BGC-MAC scenario and the raw BGC baseline. This adds BGCs that have
protein sequences and class labels but no usable product SMILES:

```bash
python -m projects.mibig_bgc_np.scripts.cache_features \
  --data_dir data/MIBIG/processed \
  --out_dir cache/ohe \
  --config projects/mibig_bgc_np/configs/ohe.yaml \
  --bgcmac_splits_path data/MIBIG/splits/bgcmac_fold.csv \
  --fasta_path data/MIBIG/mibig_prot_seqs_4.0.fasta
```

### BGC-MAC Run Command

```bash
python -m projects.mibig_bgc_np.scripts.run_bgcmac_classification \
  --data_dir data/MIBIG/processed \
  --cache_dir cache/ohe \
  --config projects/mibig_bgc_np/configs/ohe.yaml \
  --bgcmac_splits_path data/MIBIG/splits/bgcmac_fold.csv \
  --seed 42
```

### BGC-MAC Scenarios Written By This Run

The same command writes three classification outputs:

1. **Strict CLIP-covered BGC-MAC**
   - CLIP training uses valid BGC-NP pairs from `data/MIBIG/processed/mibig_pairs.tsv`.
   - The downstream class classifier is trained and tested only on BGCs that
     appear in valid BGC-NP pairs.
   - BGCs without usable product SMILES are excluded from this scenario.
   - Output: `results/bgcmac_ensemble/ensemble_downstream_metrics.json`

2. **Full-BGC BGC-MAC**
   - CLIP training is identical to the strict scenario and still uses only valid
     BGC-NP pairs.
   - The downstream class classifier is trained and tested on all BGCs in
     `data/MIBIG/splits/bgcmac_fold.csv`.
   - Fold-10 test BGCs without molecule pairs are included because BGC-MAC
     classification only needs BGC features at downstream evaluation time.
   - Output: `results/bgcmac_ensemble/ensemble_downstream_full_bgcmac_metrics.json`

3. **Raw BGC baseline**
   - CLIP is skipped.
   - The same class head is trained directly on cached BGC encoder features for
     all BGCs in `bgcmac_fold.csv`.
   - This is OHE+MLP if the cache was built with OHE BGC features.
   - Output: `results/bgcmac_ensemble/raw_bgc_baseline/raw_bgc_baseline_summary.json`

### BGC-MAC Plots And Metrics Table

When plot saving is enabled, the BGC-MAC runner writes ROC plots, confusion
matrices, and a comparison table under:

```text
results/bgcmac_ensemble/bgcmac_benchmark_artifacts/
```

Important files:

```text
bgcmac_metrics_table.csv
bgcmac_metrics_table.png
bgcmac_strict_roc.png
bgcmac_strict_one_vs_rest_confusion_matrices.png
bgcmac_strict_expanded_confusion_matrix.png
bgcmac_full_roc.png
bgcmac_full_one_vs_rest_confusion_matrices.png
bgcmac_full_expanded_confusion_matrix.png
bgcmac_baseline_roc.png
bgcmac_baseline_one_vs_rest_confusion_matrices.png
bgcmac_baseline_expanded_confusion_matrix.png
```

The comparison table columns are:

```text
class, BGC count, model, BGC-MAC, BGC-MAC-full, baseline
```

The BGC-MAC fold procedure is:

- fold 10, marked by `is_test`, is held out as the fixed test set
- folds 1-9 are used as validation folds, one at a time
- for validation fold `i`, the remaining eight non-test folds are used for training
- nine models are trained with early stopping on validation loss
- early stopping patience defaults to 5 epochs
- only the downstream `bgc_class` task is run
- retrieval baselines for the CLIP-covered BGC-NP pairs are written under each
  member's `retrieval_baselines/` directory

Use `--patience` to change early stopping patience, `--val_folds` to run a subset
of ensemble members, or `--outdir` to choose another output directory.

## BGC-MAP Retrieval Benchmark

Use `data/MIBIG/splits/MAP_metadata_fold.csv` for the BGC-MAP explicit
BGC-product retrieval benchmark:

### BGC-MAP Cache

Build a MAP-complete cache that includes all MAP BGCs and all positive/negative
candidate products from the split file:

```bash
python -m projects.mibig_bgc_np.scripts.cache_features \
  --data_dir data/MIBIG/processed \
  --out_dir cache/mibig_map \
  --config projects/mibig_bgc_np/configs/default.yaml \
  --map_metadata_path data/MIBIG/splits/MAP_metadata_fold.csv \
  --fasta_path data/MIBIG/mibig_prot_seqs_4.0.fasta
```

The MAP cache must contain features for every BGC/product row that will be
evaluated. Do not drop missing rows for MAP, because the fold-10 retrieval
benchmark should keep the full candidate-pair table.

### BGC-MAP Run Command

```bash
python -m projects.mibig_bgc_np.scripts.run_bgcmap_retrieval \
  --data_dir data/MIBIG/processed \
  --cache_dir cache/mibig_map \
  --config projects/mibig_bgc_np/configs/default.yaml \
  --bgcmap_splits_path data/MIBIG/splits/MAP_metadata_fold.csv \
  --seed 42
```

The MAP file is pair-level and includes both positive and negative candidate
pairs:

- `BGC_number`: query BGC
- `product`: candidate natural product identifier/SMILES
- `biosyn_class`: one or more BGC classes used for stratified reporting
- `is_product`: binary label for the candidate pair
- `fold`: fold assignment

This follows the BGC-MAP fold procedure:

- fold 10 is held out as the fixed test set
- folds 1-9 are used as validation folds, one at a time
- for validation fold `i`, the remaining eight non-test folds are used for training
- nine models are trained with early stopping on validation loss
- training uses only positive `is_product == 1` rows
- each model's decision threshold is selected on its own validation fold, then
  thresholds are averaged by BGC class for fold-10 confusion matrices
- fold-10 MAP rows, including negatives, are scored directly as explicit
  BGC-product pairs
- outputs demonstrate retrieval AUROC, raw-count confusion matrices, and the
  BGC-MAP metrics table
- bidirectional retrieval baselines are written under each validation member's
  `retrieval_baselines/` directory

This means the BGC-MAP confusion matrices count MAP candidate rows, not the full
BGC x compound retrieval matrix.

Outputs are written under:

```text
results/bgcmap_retrieval/
```

Important BGC-MAP outputs:

```text
ensemble_test_retrieval.json
ensemble_test_pair_scores.tsv
validation_thresholds.json
ensemble_test_bgc_class_retrieval_roc.png
ensemble_test_bgc_class_retrieval_confusion_matrices.png
ensemble_test_bgc_map_metrics_table.csv
ensemble_test_bgc_map_metrics_table.png
```

## Config Overrides

Pipeline scripts support dot-path config overrides through repeated `--override`
arguments. For example:

```bash
python -m projects.mibig_bgc_np.scripts.train_contrastive \
  --data_dir data/MIBIG/processed \
  --cache_dir cache/ohe \
  --config projects/mibig_bgc_np/configs/ohe.yaml \
  --override train.epochs=5 \
  --override output.dir=results/mibig_quick
```

## Runnable Entrypoints

```text
scripts/preprocess_mibig.py
scripts/make_mibig_splits.py
scripts/train_downstream.py
projects/mibig_bgc_np/scripts/cache_features.py
projects/mibig_bgc_np/scripts/train_contrastive.py
projects/mibig_bgc_np/scripts/eval_retrieval.py
projects/mibig_bgc_np/scripts/viz_umap.py
projects/mibig_bgc_np/scripts/train_downstream.py
projects/mibig_bgc_np/scripts/run_cv10.py
projects/mibig_bgc_np/scripts/run_bgcmac_classification.py
projects/mibig_bgc_np/scripts/run_bgcmac_ensemble.py
projects/mibig_bgc_np/scripts/run_bgcmap_retrieval.py
```
