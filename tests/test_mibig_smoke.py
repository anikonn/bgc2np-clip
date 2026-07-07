from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch

from clip_core.losses import multi_positive_infonce_loss, symmetric_infonce_loss
from mibig_clip.data.splits import (
    assign_cv_folds_by_bgc,
    assign_cv_folds_by_np,
    random_split_by_bgc,
    random_split_by_np,
)
from projects.mibig_bgc_np.data.datasets import CachedInteractionDataset, build_interactions
from projects.mibig_bgc_np.models.clip_dual import DualEncoderCLIP
from projects.mibig_bgc_np.scripts.run_bgcmac_ensemble import _build_bgcmac_interactions, _load_bgcmac_fold_table
from projects.mibig_bgc_np.training.contrastive_trainer import _build_batch_positive_mask
from projects.mibig_bgc_np.training.downstream_trainer import (
    _binary_roc_curve,
    _build_bgc_multilabel_features,
    _frame_to_tensor_dataset,
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
                "BGC_number,fold,is_test",
                "B1,1,False",
                "B2,2,False",
                "B3,10,True",
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
