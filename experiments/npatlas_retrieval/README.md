# NPAtlas retrieval experiment

This paper experiment evaluates BGC-to-NP retrieval with the frozen baseline
ESM2-domain/MolFormer CLIP checkpoints from all four 10-fold CV split
strategies (`bgc`, `np`, `combined`, and `strict`).

Two candidate libraries are evaluated for every test BGC:

1. `sampled_10000`: all matched true NPAtlas products plus random negatives up
   to 10,000 candidates (seed 42).
2. `all`: all valid rows in `data/NPAtlas_download_2024_09.tsv` (36,453 rows
   after SMILES validation; 36,420 unique canonical SMILES).

Ground truth is matched by InChIKey, with canonical SMILES as fallback. The
reported `max_maccs_tanimoto_at_100` is the maximum MACCS/Tanimoto similarity
between any of the top-100 retrieved candidates and any true product.

Submit the complete workflow with:

```bash
condor_submit_dag condor/npatlas_retrieval.dag
```

The cache job writes `cache/npatlas_molformer/`. Per-split JSON and per-query
CSV files, followed by `npatlas_retrieval_all_splits.csv`, are written to
`results/paper_plots/npatlas_retrieval/`.
