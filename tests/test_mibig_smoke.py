from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch

from clip_core.losses import multi_positive_infonce_loss, symmetric_infonce_loss
from clip_core.retrieval import _metrics_from_sorted_positive_mask
from mibig_clip.data.splits import (
    assign_cv_folds_by_bgc,
    assign_cv_folds_by_np,
    random_split_by_bgc,
    random_split_by_np,
)
from projects.mibig_bgc_np.data.datasets import CachedInteractionDataset, build_interactions
from projects.mibig_bgc_np.models.clip_dual import DualEncoderCLIP
from projects.mibig_bgc_np.scripts.run_bgcmac_ensemble import _build_bgcmac_interactions, _load_bgcmac_fold_table
from projects.mibig_bgc_np.eval.retrieval_class_metrics import _precision_recall_f1
from projects.mibig_bgc_np.eval.regression_metrics import pearson
from projects.mibig_bgc_np.scripts.plot_molecular_property_prediction import build_molecular_property_metric_table
from projects.mibig_bgc_np.training.contrastive_trainer import _build_batch_positive_mask
from projects.mibig_bgc_np.training.downstream_trainer import (
    BIOACTIVITY_CLASS_NAMES,
    NPCLASSIFIER_TASKS,
    _binary_roc_curve,
    _build_bgc_multilabel_features,
    _frame_to_tensor_dataset,
    _load_bioactivity_class_table,
    _load_npclassifier_bgc_label_table,
    _safe_molecular_properties,
)


def _write_tsv(path: Path, header: str, rows: list[str]) -> None:
    path.write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    payload = "\n".join(json.dumps(row) for row in rows) + "\n"
    path.write_text(payload, encoding="utf-8")


def test_data_join_and_forward(tmp_path: Path) -> None:
    data_dir = tmp_path / "processed"
    data_dir.mkdir(parents=True)

    _write_tsv(
        data_dir / "mibig_pairs.tsv",
        "bgc_id\tcompound_id\tsmiles\tsplit\tbgc_class",
        [
            "B1\tC1\tCCO\ttrain\tNRPS",
            "B2\tC2\tCCN\tval\tPKS",
        ],
    )
    _write_jsonl(
        data_dir / "bgc_proteins.jsonl",
        [
            {"bgc_id": "B1", "protein_ids": ["P1"], "protein_seqs": ["MKT"]},
            {"bgc_id": "B2", "protein_ids": ["P2"], "protein_seqs": ["MSS"]},
        ],
    )

    interactions = build_interactions(data_dir)
    assert len(interactions) == 2

    bgc_cache = {"B1": torch.randn(320), "B2": torch.randn(320)}
    compound_cache = {"C1": torch.randn(2048), "C2": torch.randn(2048)}
    torch.save(bgc_cache, data_dir / "bgc_features.pt")
    torch.save(compound_cache, data_dir / "compound_features.pt")

    ds = CachedInteractionDataset(
        interactions=interactions,
        bgc_cache_path=data_dir / "bgc_features.pt",
        compound_cache_path=data_dir / "compound_features.pt",
        split="train",
    )
    sample = ds[0]

    model = DualEncoderCLIP(
        bgc_input_dim=320,
        compound_input_dim=2048,
        emb_dim=64,
        hidden_dim=128,
        dropout=0.1,
    )
    loss, logits = model(
        sample["bgc_feature"].unsqueeze(0),
        sample["compound_feature"].unsqueeze(0),
    )

    assert loss.item() >= 0.0
    assert logits.shape == (1, 1)


def test_multi_positive_loss_extends_diagonal() -> None:
    logits = torch.tensor(
        [
            [3.0, 2.5, -1.0],
            [-1.0, 3.0, -1.0],
            [2.5, -1.0, 3.0],
        ]
    )
    positive_mask = torch.tensor(
        [
            [True, True, False],
            [False, True, False],
            [True, False, True],
        ]
    )

    diagonal_loss = symmetric_infonce_loss(logits)
    multi_loss = multi_positive_infonce_loss(logits, positive_mask)

    assert multi_loss < diagonal_loss


def test_batch_positive_mask_uses_all_known_train_pairs() -> None:
    mask = _build_batch_positive_mask(
        bgc_ids=["B1", "B2", "B3"],
        compound_ids=["C1", "C2", "C3"],
        positive_pairs={("B1", "C1"), ("B1", "C2"), ("B2", "C2"), ("B3", "C1"), ("B3", "C3")},
        device=torch.device("cpu"),
    )

    expected = torch.tensor(
        [
            [True, True, False],
            [False, True, False],
            [True, False, True],
        ]
    )
    assert torch.equal(mask, expected)


def test_split_types_control_leakage_constraints() -> None:
    bgc_ids = ["B1", "B2", "B3", "B4", "B5", "B6"]
    bgc_to_compounds = {
        "B1": {"C_shared_1"},
        "B2": {"C_shared_1"},
        "B3": {"C_shared_2"},
        "B4": {"C_shared_2"},
        "B5": {"C5"},
        "B6": {"C6"},
    }

    bgc_only_assignments = random_split_by_bgc(
        bgc_ids,
        seed=42,
        train_frac=0.5,
        val_frac=0.25,
        test_frac=0.25,
    )
    assert set(bgc_only_assignments) == set(bgc_ids)

    combined_assignments = random_split_by_bgc(
        bgc_ids,
        seed=42,
        train_frac=0.5,
        val_frac=0.25,
        test_frac=0.25,
        bgc_to_compound_ids=bgc_to_compounds,
    )
    assert combined_assignments["B1"] == combined_assignments["B2"]
    assert combined_assignments["B3"] == combined_assignments["B4"]

    np_assignments = random_split_by_np(
        bgc_to_compound_ids={
            "B1": {"C1", "C2"},
            "B2": {"C2"},
            "B3": {"C3"},
        },
        seed=42,
        train_frac=0.34,
        val_frac=0.33,
        test_frac=0.33,
    )
    assert np_assignments[("B1", "C2")].split == np_assignments[("B2", "C2")].split
    assert {assignment.split for assignment in np_assignments.values()} == {"train", "val", "test"}

    cv_assignments = assign_cv_folds_by_bgc(
        bgc_ids,
        seed=42,
        n_folds=3,
        bgc_to_compound_ids=bgc_to_compounds,
    )
    assert cv_assignments["B1"].fold_id == cv_assignments["B2"].fold_id
    assert cv_assignments["B3"].fold_id == cv_assignments["B4"].fold_id

    np_cv_assignments = assign_cv_folds_by_np(
        bgc_to_compound_ids={
            "B1": {"C1", "C2"},
            "B2": {"C2"},
            "B3": {"C3"},
        },
        seed=42,
        n_folds=3,
    )
    assert np_cv_assignments[("B1", "C2")].fold_id == np_cv_assignments[("B2", "C2")].fold_id


def test_build_interactions_supports_pair_level_split_files(tmp_path: Path) -> None:
    data_dir = tmp_path / "processed"
    data_dir.mkdir(parents=True)
    _write_tsv(
        data_dir / "mibig_pairs.tsv",
        "bgc_id\tcompound_id\tsmiles\tbgc_class",
        [
            "B1\tC1\tCCO\tNRPS",
            "B1\tC2\tCCN\tNRPS",
            "B2\tC2\tCCN\tPKS",
        ],
    )
    _write_jsonl(
        data_dir / "bgc_proteins.jsonl",
        [
            {"bgc_id": "B1", "protein_ids": ["P1"], "protein_seqs": ["MKT"]},
            {"bgc_id": "B2", "protein_ids": ["P2"], "protein_seqs": ["MSS"]},
        ],
    )
    split_path = tmp_path / "np_random.tsv"
    _write_tsv(
        split_path,
        "bgc_id\tcompound_id\tsplit",
        [
            "B1\tC1\ttrain",
            "B1\tC2\ttest",
            "B2\tC2\ttest",
        ],
    )

    interactions = build_interactions(data_dir, splits_path=split_path)

    split_by_pair = {
        (row.bgc_id, row.compound_id): row.split
        for row in interactions[["bgc_id", "compound_id", "split"]].itertuples(index=False)
    }
    assert split_by_pair == {
        ("B1", "C1"): "train",
        ("B1", "C2"): "test",
        ("B2", "C2"): "test",
    }


def test_build_interactions_maps_fold_id_files_to_test_val_train(tmp_path: Path) -> None:
    data_dir = tmp_path / "processed"
    data_dir.mkdir(parents=True)
    _write_tsv(
        data_dir / "mibig_pairs.tsv",
        "bgc_id\tcompound_id\tsmiles\tbgc_class",
        [
            "B1\tC1\tCCO\tNRPS",
            "B2\tC2\tCCN\tPKS",
            "B3\tC3\tCCC\tRiPP",
            "B4\tC4\tCCCl\tTerpene",
        ],
    )
    _write_jsonl(
        data_dir / "bgc_proteins.jsonl",
        [
            {"bgc_id": "B1", "protein_ids": ["P1"], "protein_seqs": ["MKT"]},
            {"bgc_id": "B2", "protein_ids": ["P2"], "protein_seqs": ["MSS"]},
            {"bgc_id": "B3", "protein_ids": ["P3"], "protein_seqs": ["MAA"]},
            {"bgc_id": "B4", "protein_ids": ["P4"], "protein_seqs": ["MNN"]},
        ],
    )
    split_path = tmp_path / "cv.tsv"
    _write_tsv(
        split_path,
        "bgc_id\tfold_id",
        [
            "B1\t1",
            "B2\t2",
            "B3\t3",
            "B4\t4",
        ],
    )

    interactions = build_interactions(data_dir, splits_path=split_path, cv_fold=2, val_fold=3)

    split_by_bgc = {
        row.bgc_id: row.split
        for row in interactions[["bgc_id", "split"]].drop_duplicates().itertuples(index=False)
    }
    assert split_by_bgc == {"B1": "train", "B2": "test", "B3": "val", "B4": "train"}


def test_bgcmac_fold_table_assigns_fixed_test_and_rotating_val(tmp_path: Path) -> None:
    data_dir = tmp_path / "processed"
    data_dir.mkdir(parents=True)
    _write_tsv(
        data_dir / "mibig_pairs.tsv",
        "bgc_id\tcompound_id\tsmiles\tbgc_class",
        [
            "B1\tC1\tCCO\tNRPS",
            "B2\tC2\tCCN\tPKS",
            "B3\tC3\tCCC\tRiPP",
        ],
    )
    _write_jsonl(
        data_dir / "bgc_proteins.jsonl",
        [
            {"bgc_id": "B1", "protein_ids": ["P1"], "protein_seqs": ["MKT"]},
            {"bgc_id": "B2", "protein_ids": ["P2"], "protein_seqs": ["MSS"]},
            {"bgc_id": "B3", "protein_ids": ["P3"], "protein_seqs": ["MAA"]},
        ],
    )
    split_path = tmp_path / "bgcmac_fold.csv"
    split_path.write_text(
        "\n".join(
            [
                "BGC_number,biosyn_class,fold,is_test",
                "B1,NRPS,1,False",
                "B2,PKS,2,False",
                "B3,RiPP,10,True",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    fold_table = _load_bgcmac_fold_table(split_path, test_fold=10)
    interactions = _build_bgcmac_interactions(data_dir, fold_table, val_fold=1)

    split_by_bgc = {
        row.bgc_id: row.split
        for row in interactions[["bgc_id", "split"]].drop_duplicates().itertuples(index=False)
    }
    assert split_by_bgc == {"B1": "val", "B2": "train", "B3": "test"}


def test_empty_bgc_multilabel_features_have_expected_shape() -> None:
    model = DualEncoderCLIP(
        bgc_input_dim=320,
        compound_input_dim=2048,
        emb_dim=64,
        hidden_dim=128,
        dropout=0.1,
    )

    x, y = _build_bgc_multilabel_features(
        bgc_df=pd.DataFrame(),
        model=model,
        bgc_cache={},
        label_to_idx={"NRPS": 0, "PKS": 1},
        device=torch.device("cpu"),
        batch_size=16,
    )

    assert x.shape == (0, 64)
    assert y.shape == (0, 2)


def test_binary_roc_curve_uses_available_numpy_auc() -> None:
    fpr, tpr, auc = _binary_roc_curve(
        torch.tensor([0, 0, 1, 1]),
        torch.tensor([0.1, 0.4, 0.35, 0.8]),
    )

    assert fpr[0] == 0.0
    assert tpr[0] == 0.0
    assert auc == 0.75


def test_frame_to_tensor_dataset_allows_empty_split() -> None:
    x, y = _frame_to_tensor_dataset(
        pd.DataFrame(columns=["compound_id", "compound_molecular_weight"]),
        {"C1": torch.randn(64)},
        "compound_molecular_weight",
        torch.float32,
    )

    assert x.shape == (0, 64)
    assert y.shape == (0,)


def test_bioactivity_class_table_keeps_only_selected_observed_labels(tmp_path: Path) -> None:
    data_dir = tmp_path / "processed"
    data_dir.mkdir(parents=True)
    _write_tsv(
        data_dir / "mibig_pairs.tsv",
        "bgc_id\tcompound_id\tsmiles\tbgc_class",
        [
            "B1\tC1\tCCO\tNRPS",
            "B2\tC2\tCCN\tPKS",
            "B3\tC3\tCCC\tRiPP",
        ],
    )
    _write_jsonl(
        data_dir / "bgc_proteins.jsonl",
        [
            {"bgc_id": "B1", "protein_ids": ["P1"], "protein_seqs": ["MKT"]},
            {"bgc_id": "B2", "protein_ids": ["P2"], "protein_seqs": ["MSS"]},
            {"bgc_id": "B3", "protein_ids": ["P3"], "protein_seqs": ["MAA"]},
        ],
    )
    split_path = tmp_path / "cv.tsv"
    _write_tsv(
        split_path,
        "bgc_id\tfold_id",
        [
            "B1\t1",
            "B2\t2",
            "B3\t3",
        ],
    )
    bioactivity_path = tmp_path / "bioactivity.csv"
    bioactivity_path.write_text(
        "\n".join(
            [
                "bgc_id,n_observed_bioactivities,observed_bioactivities",
                "B1,2,antibacterial;cytotoxic",
                "B2,1,other",
                "B3,1,antiviral",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    table = _load_bioactivity_class_table(
        data_dir=data_dir,
        bioactivity_table_path=bioactivity_path,
        splits_path=split_path,
        cv_fold=2,
        val_fold=3,
    )

    assert set(table["bgc_id"]) == {"B1", "B3"}
    labels_by_bgc = {row.bgc_id: row.bgc_class_list for row in table.itertuples(index=False)}
    assert labels_by_bgc["B1"] == ["antibacterial", "cytotoxic"]
    assert labels_by_bgc["B3"] == ["antiviral"]
    assert set(labels_by_bgc["B1"]).issubset(set(BIOACTIVITY_CLASS_NAMES))


def test_npclassifier_superclass_table_filters_labels_and_respects_pair_folds(tmp_path: Path) -> None:
    data_dir = tmp_path / "processed"
    data_dir.mkdir(parents=True)
    _write_tsv(
        data_dir / "mibig_pairs.tsv",
        "bgc_id\tcompound_id\tsmiles\tbgc_class",
        [
            "B1\tC1\tCCO\tNRPS",
            "B1\tC2\tCCN\tNRPS",
            "B2\tC3\tCCC\tPKS",
        ],
    )
    _write_jsonl(
        data_dir / "bgc_proteins.jsonl",
        [
            {"bgc_id": "B1", "protein_ids": ["P1"], "protein_seqs": ["MKT"]},
            {"bgc_id": "B2", "protein_ids": ["P2"], "protein_seqs": ["MSS"]},
        ],
    )
    split_path = tmp_path / "np_cv.tsv"
    _write_tsv(
        split_path,
        "bgc_id\tcompound_id\tfold_id",
        [
            "B1\tC1\t1",
            "B1\tC2\t2",
            "B2\tC3\t3",
        ],
    )
    labels_path = tmp_path / "mibig_pairs_npclassifier_labels.tsv"
    _write_tsv(
        labels_path,
        "bgc_id\tcompound_id\tcanonical_smiles\tnpclassifier_pathway\tnpclassifier_superclass\tnpclassifier_class",
        [
            "B1\tC1\tCCO\tAmino acids and Peptides\tOligopeptides\tCyclic peptides",
            "B1\tC2\tCCN\tPolyketides\tMacrolides\tDepsipeptides",
            "B2\tC3\tCCC\tAmino acids and Peptides\tOligopeptides\tLinear peptides",
        ],
    )
    counts_path = tmp_path / "npclassifier_superclass_counts.csv"
    counts_path.write_text(
        "label,n_compounds\nOligopeptides,101\nMacrolides,100\n",
        encoding="utf-8",
    )

    original_counts_path = NPCLASSIFIER_TASKS["npclassifier_superclass"]["counts_path"]
    NPCLASSIFIER_TASKS["npclassifier_superclass"]["counts_path"] = str(counts_path)
    try:
        table, labels, stats = _load_npclassifier_bgc_label_table(
            data_dir=data_dir,
            npclassifier_pair_labels_path=labels_path,
            task_name="npclassifier_superclass",
            splits_path=split_path,
            cv_fold=2,
            val_fold=3,
        )
    finally:
        NPCLASSIFIER_TASKS["npclassifier_superclass"]["counts_path"] = original_counts_path

    assert labels == ["Oligopeptides"]
    assert stats["min_count_exclusive"] == 100
    assert set(table["split"]) == {"train", "val"}
    labels_by_bgc_split = {(row.bgc_id, row.split): row.bgc_class_list for row in table.itertuples(index=False)}
    assert labels_by_bgc_split[("B1", "train")] == ["Oligopeptides"]
    assert ("B1", "test") not in labels_by_bgc_split
    assert labels_by_bgc_split[("B2", "val")] == ["Oligopeptides"]


def test_rdkit_molecular_properties_include_logp_and_tpsa() -> None:
    from rdkit import Chem

    props = _safe_molecular_properties("CCO", Chem)

    assert props["compound_rdkit_mw"] is not None
    assert props["compound_logp"] is not None
    assert props["compound_tpsa"] is not None
    assert props["compound_tpsa"] > 0.0


def test_pearson_handles_constant_vectors() -> None:
    assert pearson(torch.ones(3).numpy(), torch.arange(3).numpy()) == 0.0
    assert abs(pearson(torch.arange(3).numpy(), torch.arange(3).numpy()) - 1.0) < 1e-12


def test_molecular_property_metric_table_reads_cv_summary(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "aggregate": {
                    "downstream": {
                        "compound_mw": {"test": {"pearson": {"mean": 0.1, "std": 0.01, "n": 10}}},
                        "compound_logp": {"test": {"pearson": {"mean": 0.2, "std": 0.02, "n": 10}}},
                        "compound_tpsa": {"test": {"spearman": {"mean": 0.3, "std": 0.03, "n": 10}}},
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    table = build_molecular_property_metric_table({"bgc": summary_path})

    row = table[(table["split"] == "bgc") & (table["property"] == "TPSA") & (table["metric"] == "spearman")]
    assert float(row.iloc[0]["mean"]) == 0.3


def test_binary_confusion_metrics_include_accuracy() -> None:
    metrics = _precision_recall_f1(
        {
            "raw": {
                "Negative": {"Negative": 8, "Positive": 2},
                "Positive": {"Negative": 1, "Positive": 9},
            }
        }
    )

    assert metrics["accuracy"] == 0.85
    assert metrics["precision"] == 9 / 11
    assert metrics["recall"] == 0.9


def test_retrieval_hit_and_recall_use_different_denominators() -> None:
    sorted_pos = torch.tensor(
        [
            [True, True, False],
            [False, False, True],
        ]
    )

    metrics = _metrics_from_sorted_positive_mask(sorted_pos)

    assert metrics.hit_at_1 == 0.5
    assert metrics.recall_at_1 == 1 / 3
    assert metrics.hit_at_5 == 1.0
    assert metrics.recall_at_5 == 1.0
