from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_INPUT_PATH = Path("/nethome/akolchina/Combi/data/MIBIG/processed/mibig_pairs.tsv")


def _require_rdkit():
    try:
        from rdkit import Chem
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "RDKit is required to compute InChIKeys. Install it first, for example with "
            "`conda install -c conda-forge rdkit`."
        ) from exc
    return Chem


def _smiles_to_inchikey(smiles: str | float | None, chem_module) -> str | None:
    if smiles is None or pd.isna(smiles):
        return None
    text = str(smiles).strip()
    if not text:
        return None
    mol = chem_module.MolFromSmiles(text)
    if mol is None:
        return None
    return str(chem_module.MolToInchiKey(mol))


def add_inchikey_column(input_path: Path, output_path: Path, smiles_column: str = "smiles") -> None:
    chem = _require_rdkit()
    df = pd.read_csv(input_path, sep="\t")

    if smiles_column not in df.columns:
        raise ValueError(f"Column '{smiles_column}' not found in {input_path}.")

    df["inchikey"] = [_smiles_to_inchikey(smiles, chem) for smiles in df[smiles_column].tolist()]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, sep="\t", index=False)

    total = len(df)
    extracted = int(df["inchikey"].notna().sum())
    missing = total - extracted
    print(f"Wrote {output_path}")
    print(f"Rows: {total}")
    print(f"InChIKeys extracted: {extracted}")
    print(f"Missing/invalid SMILES: {missing}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add an inchikey column to a TSV file using RDKit.")
    parser.add_argument("--input_path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output_path", type=Path, default=None)
    parser.add_argument("--smiles_column", type=str, default="smiles")
    parser.add_argument(
        "--in_place",
        action="store_true",
        help="Overwrite the input file. By default, writes next to it as <name>.with_inchikey.tsv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input_path.resolve()
    if args.in_place:
        output_path = input_path
    elif args.output_path is not None:
        output_path = args.output_path.resolve()
    else:
        output_path = input_path.with_name(f"{input_path.stem}.with_inchikey{input_path.suffix}")

    add_inchikey_column(
        input_path=input_path,
        output_path=output_path,
        smiles_column=args.smiles_column,
    )


if __name__ == "__main__":
    main()
