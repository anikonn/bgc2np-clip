from __future__ import annotations

from pathlib import Path

from scripts.extract_antismash_domains import extract_domain_record


def test_extracts_domains_and_retains_unsplit_cds(tmp_path: Path) -> None:
    gbk = tmp_path / "BGC0000001.gbk"
    gbk.write_text(
        """FEATURES             Location/Qualifiers
     CDS             1..300
                     /protein_id="protein_one"
                     /translation="MMMM"
     aSDomain        31..90
                     /aSDomain="PKS_KS"
                     /domain_id="domain_one"
                     /protein_start="10"
                     /protein_end="30"
                     /translation="AAAA"
     aSDomain        121..180
                     /aSDomain="PKS_AT"
                     /domain_id="domain_two"
                     /protein_start="40"
                     /protein_end="60"
                     /translation="CCCC"
     gene            complement(400..600)
     CDS             complement(400..600)
                     /protein_id="protein_two"
                     /translation="DD
                     DD"
ORIGIN
//
""",
        encoding="utf-8",
    )

    record = extract_domain_record(gbk)

    assert record["bgc_id"] == "BGC0000001"
    assert record["protein_ids"] == ["domain_one", "domain_two", "protein_two"]
    assert record["protein_seqs"] == ["AAAA", "CCCC", "DDDD"]
    assert record["sequence_sources"] == [
        "antismash_domain",
        "antismash_domain",
        "unsplit_cds",
    ]
    assert record["parent_cds_indices"] == [0, 0, 1]
    assert record["item_genomic_locations"] == ["31..90", "121..180", "complement(400..600)"]
    assert record["item_protein_starts"] == [10, 40, 0]
    assert record["item_protein_ends"] == [30, 60, 4]
    assert record["parent_cds_locations"] == ["1..300", "1..300", "complement(400..600)"]
    assert record["n_cds"] == 2
    assert record["n_antismash_domains"] == 2
    assert record["n_emitted_sequences"] == 3
