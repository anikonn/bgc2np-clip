from __future__ import annotations

from pathlib import Path

import pandas as pd


def resolve_seed_tables(data_dir: str | Path) -> tuple[Path, Path, Path]:
    """Resolve interS, lig, prot files from either flat or DataSAIL-like layout."""
    seed_path = Path(data_dir)
    direct = (
        seed_path / "interS.tsv",
        seed_path / "lig.tsv",
        seed_path / "prot.tsv",
    )
    if all(p.exists() for p in direct):
        return direct

    nested_base = seed_path / "kiba" / "resources" / "tables"
    nested = (
        nested_base / "interS.tsv",
        nested_base / "lig.tsv",
        nested_base / "prot.tsv",
    )
    if all(p.exists() for p in nested):
        return nested

    raise FileNotFoundError(
        "Could not find interS.tsv, lig.tsv, prot.tsv in data_dir. "
        f"Checked: {direct} and {nested}"
    )


def maybe_to_parquet(df: pd.DataFrame, parquet_path: Path) -> bool:
    """Try writing parquet; return False if backend is missing."""
    try:
        df.to_parquet(parquet_path, index=False)
        return True
    except Exception:
        return False
