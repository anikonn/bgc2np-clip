from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


FEATURE_KEY_START = 5
FEATURE_KEY_END = 21


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create BiG-SCAPE GBK inputs with synthetic cand_cluster features copied from region features."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("data/MIBIG/mibig_gbk_bigscape_input"),
        help="Current BiG-SCAPE GBK input folder.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/MIBIG/mibig_gbk_bigscape_input_cand_cluster"),
        help="Output folder for patched GBKs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace output_dir if it already exists.",
    )
    return parser.parse_args()


def _feature_key(line: str) -> str | None:
    if not line.startswith("     ") or len(line) < FEATURE_KEY_END:
        return None
    key = line[FEATURE_KEY_START:FEATURE_KEY_END].strip()
    return key or None


def _feature_location(line: str) -> str:
    return line[FEATURE_KEY_END:].rstrip("\n")


def _qualifier_value(block: list[str], name: str) -> str | None:
    prefix = f"/{name}="
    for line in block[1:]:
        value = line[FEATURE_KEY_END:].strip()
        if not value.startswith(prefix):
            continue
        return value[len(prefix) :].strip()
    return None


def _region_qualifiers(block: list[str]) -> list[str]:
    keep_prefixes = (
        "/contig_edge=",
        "/product=",
        "/region_number=",
        "/rules=",
        "/detection_rule=",
        "/tool=",
    )
    qualifiers: list[str] = []
    for line in block[1:]:
        value = line[FEATURE_KEY_END:].rstrip("\n")
        if value.lstrip().startswith(keep_prefixes):
            qualifiers.append(line)
    return qualifiers


def _has_qualifier(block: list[str], name: str) -> bool:
    prefix = f"/{name}="
    return any(line[FEATURE_KEY_END:].strip().startswith(prefix) for line in block[1:])


def _add_region_candidate_cluster_number(region_block: list[str], cluster_number: int) -> list[str]:
    if _has_qualifier(region_block, "candidate_cluster_numbers"):
        return list(region_block)
    output: list[str] = []
    inserted = False
    for line in region_block:
        output.append(line)
        if not inserted and line[FEATURE_KEY_END:].strip().startswith("/region_number="):
            output.append(f'                     /candidate_cluster_numbers="{cluster_number}"\n')
            inserted = True
    if not inserted:
        output.append(f'                     /candidate_cluster_numbers="{cluster_number}"\n')
    return output


def _make_proto_core_block(region_block: list[str], cluster_number: int) -> list[str]:
    location = _feature_location(region_block[0])
    block = [f"     proto_core      {location}\n"]
    block.append(f'                     /protocluster_number="{cluster_number}"\n')
    for qualifier in _region_qualifiers(region_block):
        qualifier_text = qualifier[FEATURE_KEY_END:].lstrip()
        if qualifier_text.startswith(("/region_number=", "/subregion_numbers=")):
            continue
        block.append(qualifier)
    return block


def _make_proto_cluster_block(region_block: list[str], cluster_number: int) -> list[str]:
    location = _feature_location(region_block[0])
    block = [f"     protocluster    {location}\n"]
    block.append(f'                     /protocluster_number="{cluster_number}"\n')
    for qualifier in _region_qualifiers(region_block):
        qualifier_text = qualifier[FEATURE_KEY_END:].lstrip()
        if qualifier_text.startswith(("/region_number=", "/subregion_numbers=")):
            continue
        block.append(qualifier)
    return block


def _make_cand_cluster_block(region_block: list[str], region_index: int) -> list[str]:
    location = _feature_location(region_block[0])
    region_number = _qualifier_value(region_block, "region_number") or f'"{region_index}"'
    cluster_number = int(region_number.strip('"'))
    block = [f"     cand_cluster    {location}\n"]
    block.append(f"                     /candidate_cluster_number={region_number}\n")
    block.append('                     /kind="single"\n')
    block.append(f'                     /protoclusters="{cluster_number}"\n')
    seen = {"/candidate_cluster_number=", "/kind=", "/protoclusters="}
    for qualifier in _region_qualifiers(region_block):
        qualifier_text = qualifier[FEATURE_KEY_END:].lstrip()
        qualifier_name = qualifier_text.split("=", 1)[0] + "=" if "=" in qualifier_text else qualifier_text
        if qualifier_name in seen or qualifier_text.startswith(("/region_number=", "/subregion_numbers=")):
            continue
        seen.add(qualifier_name)
        block.append(qualifier)
    return block


def _existing_cand_cluster_locations(lines: list[str]) -> set[str]:
    locations: set[str] = set()
    for line in lines:
        if _feature_key(line) == "cand_cluster":
            locations.add(_feature_location(line).strip())
    return locations


def _existing_feature_locations(lines: list[str], feature_name: str) -> set[str]:
    locations: set[str] = set()
    for line in lines:
        if _feature_key(line) == feature_name:
            locations.add(_feature_location(line).strip())
    return locations


def _patch_gbk(lines: list[str]) -> tuple[list[str], dict[str, Any]]:
    existing_cand_locations = _existing_cand_cluster_locations(lines)
    existing_proto_cluster_locations = _existing_feature_locations(lines, "protocluster")
    existing_proto_core_locations = _existing_feature_locations(lines, "proto_core")
    output: list[str] = []
    region_count = 0
    added_count = 0
    existing_cand_count = len(existing_cand_locations)
    existing_proto_cluster_count = len(existing_proto_cluster_locations)
    existing_proto_core_count = len(existing_proto_core_locations)
    added_proto_cluster_count = 0
    added_proto_core_count = 0
    added_region_parent_count = 0
    i = 0
    while i < len(lines):
        line = lines[i]
        key = _feature_key(line)
        if key != "region":
            output.append(line)
            i += 1
            continue

        region_count += 1
        block = [line]
        j = i + 1
        while j < len(lines):
            next_key = _feature_key(lines[j])
            if next_key is not None or lines[j].startswith("ORIGIN"):
                break
            block.append(lines[j])
            j += 1

        cluster_number = region_count
        explicit_region_number = _qualifier_value(block, "region_number")
        if explicit_region_number is not None:
            cluster_number = int(explicit_region_number.strip('"'))
        patched_region_block = _add_region_candidate_cluster_number(block, cluster_number)
        if len(patched_region_block) != len(block):
            added_region_parent_count += 1
        output.extend(patched_region_block)
        location_key = _feature_location(line).strip()
        if location_key not in existing_proto_core_locations:
            output.extend(_make_proto_core_block(block, cluster_number))
            existing_proto_core_locations.add(location_key)
            added_proto_core_count += 1
        if location_key not in existing_proto_cluster_locations:
            output.extend(_make_proto_cluster_block(block, cluster_number))
            existing_proto_cluster_locations.add(location_key)
            added_proto_cluster_count += 1
        if location_key not in existing_cand_locations:
            output.extend(_make_cand_cluster_block(block, region_count))
            existing_cand_locations.add(location_key)
            added_count += 1
        i = j

    final_cand_count = existing_cand_count + added_count
    return output, {
        "region_count": region_count,
        "added_region_candidate_cluster_numbers": added_region_parent_count,
        "existing_cand_cluster_count": existing_cand_count,
        "added_cand_cluster_count": added_count,
        "final_cand_cluster_count": final_cand_count,
        "existing_protocluster_count": existing_proto_cluster_count,
        "added_protocluster_count": added_proto_cluster_count,
        "final_protocluster_count": existing_proto_cluster_count + added_proto_cluster_count,
        "existing_proto_core_count": existing_proto_core_count,
        "added_proto_core_count": added_proto_core_count,
        "final_proto_core_count": existing_proto_core_count + added_proto_core_count,
    }


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")
    if output_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"Output directory already exists: {output_dir}. Use --overwrite to replace it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    gbk_paths = sorted(input_dir.glob("*.gbk"))
    file_reports: list[dict[str, Any]] = []
    examples_unpatched: list[dict[str, Any]] = []
    n_with_region = 0
    n_with_cand_after = 0

    for path in gbk_paths:
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        patched_lines, stats = _patch_gbk(lines)
        out_path = output_dir / path.name
        out_path.write_text("".join(patched_lines), encoding="utf-8")

        has_region = stats["region_count"] > 0
        has_cand_after = stats["final_cand_cluster_count"] > 0
        if has_region:
            n_with_region += 1
        if has_cand_after:
            n_with_cand_after += 1
        if not has_region or not has_cand_after:
            examples_unpatched.append(
                {
                    "file": path.name,
                    "reason": "no_region" if not has_region else "no_cand_cluster_after_patch",
                    **stats,
                }
            )
        file_reports.append({"file": path.name, **stats})

    report: dict[str, Any] = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "input_gbks": len(gbk_paths),
        "output_gbks": len(list(output_dir.glob("*.gbk"))),
        "files_with_at_least_one_region": n_with_region,
        "files_with_at_least_one_cand_cluster_after_patching": n_with_cand_after,
        "files_without_region_or_cand_cluster_after_patch": len(examples_unpatched),
        "examples_files_that_could_not_be_patched": examples_unpatched[:20],
        "total_region_features": int(sum(item["region_count"] for item in file_reports)),
        "total_added_region_candidate_cluster_numbers": int(
            sum(item["added_region_candidate_cluster_numbers"] for item in file_reports)
        ),
        "total_existing_cand_cluster_features": int(sum(item["existing_cand_cluster_count"] for item in file_reports)),
        "total_added_cand_cluster_features": int(sum(item["added_cand_cluster_count"] for item in file_reports)),
        "total_final_cand_cluster_features": int(sum(item["final_cand_cluster_count"] for item in file_reports)),
        "total_existing_protocluster_features": int(sum(item["existing_protocluster_count"] for item in file_reports)),
        "total_added_protocluster_features": int(sum(item["added_protocluster_count"] for item in file_reports)),
        "total_final_protocluster_features": int(sum(item["final_protocluster_count"] for item in file_reports)),
        "total_existing_proto_core_features": int(sum(item["existing_proto_core_count"] for item in file_reports)),
        "total_added_proto_core_features": int(sum(item["added_proto_core_count"] for item in file_reports)),
        "total_final_proto_core_features": int(sum(item["final_proto_core_count"] for item in file_reports)),
    }
    report_path = output_dir / "patch_cand_cluster_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
