# Rank-wise top-10 structural similarity

For every strict-CV10 test BGC and each retrieved position 1 through 10, this
analysis computes the maximum MACCS/Tanimoto similarity between the candidate
at that exact position and any true product of the query BGC. Values are
averaged per position over queries and folds. Shaded plot regions show one
standard deviation across the ten fold means.

Compared methods are the baseline BGC2NP-CLIP model and the four retrieval
baselines stored by the standard pipeline: random ranking (10 trials), frozen
encoder similarity, kNN transfer, and a trained linear projection.

Reproduce with:

```bash
python -m projects.mibig_bgc_np.scripts.plot_rankwise_tanimoto
```

Outputs are `results/paper_plots/strict_rankwise_top10_maccs.{png,pdf}` plus
summary and per-query CSV files with the same filename prefix.
