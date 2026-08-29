# Encoder fine-tuning screening experiment

Status: **intermediate / not part of the final paper model unless promoted after
review**.

## Scientific question

Can selective or full fine-tuning of ESM2-t30 and/or MolFormer improve strict
BGC-NP retrieval without changing the canonical hierarchical mean BGC
aggregation?

## Controlled representation

- BGC: antiSMASH domain sequence -> ESM2 amino-acid-token mean -> domain mean
  within protein -> protein mean within BGC.
- NP: MolFormer pooled SMILES representation.
- Both sides retain the existing CLIP projection heads and multi-positive
  contrastive loss.
- Online batches use a 256-pair detached cross-batch queue. Gradient
  accumulation alone does not increase the number of CLIP negatives; the queue
  restores a broad candidate set while keeping domain-token memory manageable.
- Only train-fold entities contribute gradients. Validation selects the epoch;
  test is evaluated once from the selected trainable delta.

## Screening matrix

All screening runs use strict folds 1, 2, and 3.

| Run | ESM2 | MolFormer |
|---|---|---|
| `b0_frozen_online` | frozen | frozen |
| `e2_esm_last2` | last 2 blocks | frozen |
| `e5_esm_last5` | last 5 blocks | frozen |
| `ef_esm_full` | full | frozen |
| `m2_molformer_last2` | frozen | last 2 blocks |
| `m4_molformer_last4` | frozen | last 4 blocks |
| `mf_molformer_full` | frozen | full |
| `em_esm5_molformer4` | last 5 blocks | last 4 blocks |
| `ff_both_full` | full | full |

The code discovers and records the actual number of Transformer blocks and
trainable parameters for both remote checkpoints. `last N` unfreezes exactly the
last N blocks; MolFormer's separate pooler is also adapted when MolFormer is
partially unfrozen.

## Run

On the HTCondor submit host:

```bash
cd /nethome/akolchina/Combi
condor_submit_dag condor/encoder_finetuning_screen.dag
```

Nine GPU jobs run independently; the final CPU node creates:

`results/intermediate/encoder_finetuning/screen_strict_folds_1_2_3.csv`

Each fold stores only `trainable_delta_best.pt`, not a duplicate of every
pretrained parameter. Base model names and unfreezing metadata are stored in
`result.json`.

## Promotion rule

Do not move a screening run into `results/paper_plots/` automatically. Review
retrieval, fold consistency, runtime, and memory first. A promoted candidate
must then be rerun on strict CV10 and evaluated on downstream tasks.

## Publication cleanup

The entire `results/intermediate/encoder_finetuning/` tree is disposable after
the selected results and provenance have been copied into the paper experiment
registry. Condor files contain local cluster paths and are intentionally ignored
by git. The reusable model, training script, config, tests, and this experiment
description are source-controlled.
