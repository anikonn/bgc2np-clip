# Experiment registry

This file records the provenance and interpretation of experiments used in the
paper. Entries labelled **ours / self-computed** were generated in this
repository and must not be described as embeddings supplied by another model or
paper.

## 2026-08-03 — Our hierarchical antiSMASH ESM2-t30 BGC embeddings

**Status:** complete; canonical self-computed BGC representation for subsequent
experiments unless an ablation explicitly states otherwise.

**Ownership/provenance:** ours / self-computed from MIBiG antiSMASH annotations.
These are distinct from the precomputed domain embeddings obtained from the
BGC-MAC data release.

### Construction

- Source antiSMASH records: `data/MIBIG/antismash_annotation/BGC*.gbk`.
- Domain extraction: each CDS contributes all child antiSMASH `aSDomain`
  translations. A CDS without an `aSDomain` contributes its complete translated
  protein as one unsplit item.
- Coverage fallback: nine strict-split BGCs without a base antiSMASH GBK use all
  complete proteins from `data/MIBIG/processed/bgc_proteins.jsonl` as unsplit
  items: BGC0000476, BGC0000478, BGC0000482, BGC0000618, BGC0000621,
  BGC0000848, BGC0001134, BGC0002750, and BGC0002846.
- Sequence encoder: `facebook/esm2_t30_150M_UR50D`, frozen, 640 dimensions.
- Maximum input length: 1024 tokens including ESM special tokens.
- Domain/item embedding: masked mean over amino-acid token embeddings, excluding
  BOS, EOS, and padding.
- Protein embedding: mean over the domain/item embeddings assigned to that CDS.
- BGC embedding: mean over protein embeddings in the BGC.
- This is therefore hierarchical domain -> protein -> BGC mean pooling, not a
  direct unweighted mean over every domain in the BGC.

### Saved artifacts

- Cache and machine-readable provenance:
  `cache/antismash_hierarchical_esm2_t30_molformer/`
- Final BGC embeddings: `bgc_features.pt` (2,114 BGCs, 640 dimensions).
- Domain embeddings: `domain_features.pt` (ordered matrices per BGC).
- Protein embeddings: `protein_features.pt` (matrices per BGC).
- Extracted sequences and hierarchy: `antismash_domains_with_parents.jsonl`.
- Exact cache metadata: `cache_index.json`.
- Reproduction submit file: `condor/hierarchical_antismash_esm2_t30.sub`.
- Strict CV10 results:
  `results/hierarchical_esm2_t30_domains_molformer_strict_cv10/`.

Cache statistics: 2,114 BGCs, 67,350 domain-or-unsplit-protein sequences,
35,750 proteins, and 344 sequences truncated by the 1024-token ESM limit.

### Retrieval comparison with precomputed BGC-MAC embeddings

The controlled comparison used the same strict connected-component CV10 split,
MolFormer compound embeddings, CLIP training pipeline, loss, and retrieval
metrics. Only the BGC input embeddings changed.

| Direction | Metric | Ours, ESM2-t30 hierarchical | Precomputed BGC-MAC | Ours - BGC-MAC |
|---|---:|---:|---:|---:|
| BGC -> NP | MRR | 0.197 +/- 0.017 | 0.196 +/- 0.027 | +0.001 |
| BGC -> NP | Hit@1 | 0.103 +/- 0.018 | 0.099 +/- 0.024 | +0.005 |
| BGC -> NP | Hit@5 | 0.273 +/- 0.036 | 0.264 +/- 0.048 | +0.009 |
| BGC -> NP | Hit@10 | 0.382 +/- 0.034 | 0.402 +/- 0.044 | -0.020 |
| BGC -> NP | Hit@50 | 0.737 +/- 0.032 | 0.768 +/- 0.038 | -0.031 |
| BGC -> NP | Hit@100 | 0.856 +/- 0.033 | 0.900 +/- 0.026 | -0.044 |
| NP -> BGC | MRR | 0.217 +/- 0.019 | 0.230 +/- 0.033 | -0.013 |
| NP -> BGC | Hit@1 | 0.105 +/- 0.019 | 0.110 +/- 0.026 | -0.005 |
| NP -> BGC | Hit@5 | 0.323 +/- 0.043 | 0.334 +/- 0.052 | -0.010 |
| NP -> BGC | Hit@10 | 0.465 +/- 0.041 | 0.492 +/- 0.056 | -0.027 |
| NP -> BGC | Hit@50 | 0.813 +/- 0.037 | 0.870 +/- 0.024 | -0.057 |
| NP -> BGC | Hit@100 | 0.923 +/- 0.023 | 0.958 +/- 0.015 | -0.034 |

**Interpretation for the paper:** our embeddings reproduce the precomputed
BGC-MAC retrieval performance at the top of the ranking (nearly identical MRR
and Hit@1). The precomputed embeddings have a modest advantage at broader Top-K,
especially for NP -> BGC retrieval. Do not claim that the methods are exactly
equivalent or that a significance test was performed; the reported variation is
the standard deviation across ten folds.

## 2026-08-03 — Domain aggregation ablation

**Status:** complete; DAG finished all five nodes and each aggregation mode has
10/10 strict CV folds.

Using the same frozen self-computed ESM2-t30 domain/item embeddings and MolFormer
compound embeddings, we compared padding-masked flat mean, learned attention,
and a two-layer Transformer. Attention was best for retrieval: BGC -> NP MRR
0.217 versus 0.205 for mean, and NP -> BGC MRR 0.235 versus 0.220. The
Transformer was worse: 0.172 and 0.195, respectively. Attention did not improve
every downstream task, so it is recorded as a retrieval improvement rather than
a universal representation improvement.

Canonical table and full interpretation:
`results/paper_plots/domain_aggregation_ablation/REPORT.md`.

### Hierarchical-attention follow-up (completed 2026-08-04)

The 10-fold strict-CV follow-up compared attention among domains within each
protein followed by either protein mean or protein attention. Domain attention
-> protein mean produced MRR 0.192 (BGC -> NP) and 0.220 (NP -> BGC). Domain
attention -> protein attention produced 0.218 and 0.240. The latter is
effectively tied with flat attention (0.217 and 0.235), while the former is
worse. No difference reaches the project's practical threshold of 0.1.

Decision: record this as a negative ablation. Explicit hierarchical attention
does not materially improve retrieval or downstream performance and is not the
recommended main-model direction.

### Paper placement decision

Place the complete domain-aggregation experiment in the paper Appendix/Ablation
Studies rather than in the main model description. Include masked mean, flat
attention, Transformer, domain-attention -> protein-mean, and domain-attention ->
protein-attention under the same strict CV10 protocol.

The wording must be nuanced rather than saying attention simply failed:

- learned hierarchical attention produced small retrieval gains on some metrics
  but did not consistently outperform simpler aggregation enough to justify the
  added complexity;
- macro BGC-class AUROC changed from 0.926 to 0.934 (+0.008);
- NRPS AUROC changed from 0.913 to 0.941 (+0.028) and improved on all ten folds,
  which is a consistent class-specific signal worth reporting;
- saccharide AUROC decreased from 0.906 to 0.890;
- the vanilla Transformer was clearly worse than mean and attention;
- consequently, retain hierarchical mean for the main model and present the
  learned aggregators as an informative ablation/complexity trade-off.

When drafting the Appendix later, use the canonical numbers and interpretation
from `results/paper_plots/domain_aggregation_ablation/REPORT.md` and
`bgc_class_per_class_auroc_hierarchical_attention_vs_baseline.csv`.

## 2026-08-04 — Encoder fine-tuning screen

**Status:** prepared, not yet submitted; intermediate model-selection
experiment. Do not treat any variant as a paper result until the screening table
has been reviewed and a candidate has completed strict CV10.

The experiment keeps our canonical antiSMASH hierarchical mean representation
and tests frozen, last-N-block, and full fine-tuning of ESM2-t30 and MolFormer,
both separately and jointly. Screening uses strict folds 1, 2, and 3. Outputs
are isolated under `results/intermediate/encoder_finetuning/`; complete design,
run command, artifact policy, and promotion rule are documented in
`experiments/encoder_finetuning/README.md`.
