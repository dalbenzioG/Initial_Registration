#!/usr/bin/env python3
"""
Validate mask/image metadata consistency and generate signed distance maps for TRUSTED.

Validates that each mask matches its corresponding image in origin, spacing, direction.
Generates SDMs for valid pairs. Supports --fix_metadata to copy image metadata to masks
(use with caution) and --fail_on_mismatch to exit on any metadata error.

Usage:
  python utils/validate_and_generate_sdm.py \\
    --ct_img_dir /path/to/CT/images \\
    --us_img_dir /path/to/US/images \\
    --ct_mask_dir /path/to/CT/masks \\
    --us_mask_dir /path/to/US/masks \\
    --out_dir /path/to/output/sdms \\
    [--fix_metadata] [--fail_on_mismatch] [--limit N] [--skip_existing]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import SimpleITK as sitk
from tqdm import tqdm

# Ensure repo root is on path when run as script
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.registration_utils import (
    check_metadata_equality,
    collect_mask_image_pairs,
    copy_metadata_from_reference,
    load_binary_mask,
    load_image,
    mask_is_empty,
    mask_to_signed_distance_map,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def validate_pair(
    ct_mask_path: Path | None,
    us_mask_path: Path | None,
    ct_img_path: Path | None,
    us_img_path: Path | None,
    fix_metadata: bool,
    fail_on_mismatch: bool,
) -> tuple[bool, dict[str, Any]]:
    """
    Validate mask/image metadata for a kidney pair. Optionally fix mask metadata.
    Returns (ok, report_dict).
    """
    report: dict[str, Any] = {
        "ct_mask_ok": False,
        "us_mask_ok": False,
        "ct_img_exists": ct_img_path is not None and ct_img_path.exists(),
        "us_img_exists": us_img_path is not None and us_img_path.exists(),
        "ct_mask_empty": None,
        "us_mask_empty": None,
        "ct_mask_metadata_ok": None,
        "us_mask_metadata_ok": None,
        "errors": [],
        "warnings": [],
        "fixed": False,
    }
    ok = True

    if not ct_mask_path or not ct_mask_path.exists():
        report["errors"].append("CT mask missing")
        return False, report
    if not us_mask_path or not us_mask_path.exists():
        report["errors"].append("US mask missing")
        return False, report

    report["ct_mask_ok"] = True
    report["us_mask_ok"] = True

    ct_mask = load_binary_mask(ct_mask_path)
    us_mask = load_binary_mask(us_mask_path)
    report["ct_mask_empty"] = mask_is_empty(ct_mask)
    report["us_mask_empty"] = mask_is_empty(us_mask)
    if report["ct_mask_empty"] or report["us_mask_empty"]:
        report["errors"].append(
            f"Empty mask(s): CT={report['ct_mask_empty']}, US={report['us_mask_empty']}"
        )
        return False, report

    if ct_img_path and ct_img_path.exists():
        ct_img = load_image(ct_img_path)
        ct_check = check_metadata_equality(ct_mask, ct_img)
        report["ct_mask_metadata_ok"] = ct_check.ok
        if not ct_check.ok:
            msg = (
                f"CT mask/image metadata mismatch: "
                f"origin_ok={ct_check.origin_ok}, spacing_ok={ct_check.spacing_ok}, "
                f"direction_ok={ct_check.direction_ok}, size_ok={ct_check.size_ok}"
            )
            if ct_check.origin_diff:
                msg += f"; origin_diff={ct_check.origin_diff}"
            if ct_check.spacing_diff:
                msg += f"; spacing_diff={ct_check.spacing_diff}"
            report["warnings"].append(msg)
            if fix_metadata:
                ct_mask = copy_metadata_from_reference(ct_mask, ct_img)
                report["fixed"] = True
                report["warnings"].append("Copied CT image metadata to CT mask (--fix_metadata)")
            elif fail_on_mismatch:
                report["errors"].append(msg)
                ok = False
    else:
        report["warnings"].append("CT image missing; skipping CT mask metadata check")

    if us_img_path and us_img_path.exists():
        us_img = load_image(us_img_path)
        us_check = check_metadata_equality(us_mask, us_img)
        report["us_mask_metadata_ok"] = us_check.ok
        if not us_check.ok:
            msg = (
                f"US mask/image metadata mismatch: "
                f"origin_ok={us_check.origin_ok}, spacing_ok={us_check.spacing_ok}, "
                f"direction_ok={us_check.direction_ok}, size_ok={us_check.size_ok}"
            )
            if us_check.origin_diff:
                msg += f"; origin_diff={us_check.origin_diff}"
            if us_check.spacing_diff:
                msg += f"; spacing_diff={us_check.spacing_diff}"
            report["warnings"].append(msg)
            if fix_metadata:
                us_mask = copy_metadata_from_reference(us_mask, us_img)
                report["fixed"] = True
                report["warnings"].append("Copied US image metadata to US mask (--fix_metadata)")
            elif fail_on_mismatch:
                report["errors"].append(msg)
                ok = False
    else:
        report["warnings"].append("US image missing; skipping US mask metadata check")

    report["ct_mask_final"] = ct_mask
    report["us_mask_final"] = us_mask
    return ok, report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate mask/image metadata and generate SDMs for TRUSTED kidney registration"
    )
    parser.add_argument("--ct_img_dir", type=Path, required=True, help="CT image directory")
    parser.add_argument("--us_img_dir", type=Path, required=True, help="US image directory")
    parser.add_argument("--ct_mask_dir", type=Path, required=True, help="CT mask directory")
    parser.add_argument("--us_mask_dir", type=Path, required=True, help="US mask directory")
    parser.add_argument(
        "--out_dir",
        type=Path,
        required=True,
        help="Output directory for SDMs and report",
    )
    parser.add_argument(
        "--fix_metadata",
        action="store_true",
        help="Copy image metadata to mask when mismatch (use with caution)",
    )
    parser.add_argument(
        "--fail_on_mismatch",
        action="store_true",
        help="Exit with error on any metadata mismatch",
    )
    parser.add_argument(
        "--report_format",
        choices=["json", "csv"],
        default="json",
        help="Report format (default: json)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N pairs (for debugging)",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip pairs whose SDM outputs already exist (resume interrupted runs)",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    pairs = collect_mask_image_pairs(
        args.ct_mask_dir,
        args.us_mask_dir,
        args.ct_img_dir,
        args.us_img_dir,
    )
    if not pairs:
        logger.error("No mask/image pairs found")
        return 1

    if args.limit is not None:
        pairs = pairs[: args.limit]
        logger.info("Limited to first %d pairs (--limit %d)", len(pairs), args.limit)
    else:
        logger.info("Found %d pairs", len(pairs))

    results: list[dict[str, Any]] = []
    processed = 0
    failed = 0
    skipped = 0
    missing = 0
    metadata_issues = 0
    empty_masks = 0

    for pair in tqdm(pairs, desc="Validating and generating SDMs"):
        key = pair["kidney_key"]
        r: dict[str, Any] = {
            "kidney_key": key,
            "status": "unknown",
            "ct_sdm_path": None,
            "us_sdm_path": None,
            "errors": [],
            "warnings": [],
        }

        ct_sdm_path = args.out_dir / f"{key}_imgCT_sdm.nii.gz"
        us_sdm_path = args.out_dir / f"{key}_imgUS_sdm.nii.gz"
        if args.skip_existing and ct_sdm_path.exists() and us_sdm_path.exists():
            r["status"] = "skipped"
            r["ct_sdm_path"] = str(ct_sdm_path)
            r["us_sdm_path"] = str(us_sdm_path)
            skipped += 1
            results.append(r)
            continue

        ct_mask = pair.get("ct_mask_path")
        us_mask = pair.get("us_mask_path")
        ct_img = pair.get("ct_img_path")
        us_img = pair.get("us_img_path")

        if not ct_mask or not us_mask:
            r["status"] = "missing_masks"
            r["errors"].append("Missing CT or US mask")
            missing += 1
            results.append(r)
            continue

        ok, val_report = validate_pair(
            ct_mask, us_mask, ct_img, us_img, args.fix_metadata, args.fail_on_mismatch
        )
        r["warnings"] = val_report.get("warnings", [])
        r["errors"] = val_report.get("errors", [])

        if val_report.get("ct_mask_empty") or val_report.get("us_mask_empty"):
            r["status"] = "empty_mask"
            empty_masks += 1
            results.append(r)
            continue

        if not ok and args.fail_on_mismatch:
            r["status"] = "metadata_mismatch"
            metadata_issues += 1
            results.append(r)
            failed += 1
            continue

        if val_report.get("ct_mask_metadata_ok") is False or val_report.get(
            "us_mask_metadata_ok"
        ) is False:
            metadata_issues += 1

        try:
            ct_mask_img = val_report.get("ct_mask_final") or load_binary_mask(ct_mask)
            us_mask_img = val_report.get("us_mask_final") or load_binary_mask(us_mask)

            ct_sdm = mask_to_signed_distance_map(ct_mask_img)
            us_sdm = mask_to_signed_distance_map(us_mask_img)

            sitk.WriteImage(ct_sdm, str(ct_sdm_path))
            sitk.WriteImage(us_sdm, str(us_sdm_path))

            r["status"] = "success"
            r["ct_sdm_path"] = str(ct_sdm_path)
            r["us_sdm_path"] = str(us_sdm_path)
            processed += 1
        except Exception as e:
            r["status"] = "error"
            r["errors"].append(str(e))
            failed += 1

        results.append(r)

    summary = {
        "total_pairs": len(pairs),
        "processed": processed,
        "skipped": skipped,
        "failed": failed,
        "missing_masks": missing,
        "empty_masks": empty_masks,
        "metadata_issues": metadata_issues,
    }

    report_path = args.out_dir / "validation_report.json"
    if args.report_format == "json":
        with open(report_path, "w") as f:
            json.dump(
                {"summary": summary, "results": results},
                f,
                indent=2,
            )
    else:
        import csv

        report_path = args.out_dir / "validation_report.csv"
        with open(report_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "kidney_key",
                    "status",
                    "ct_sdm_path",
                    "us_sdm_path",
                    "errors",
                    "warnings",
                ],
            )
            writer.writeheader()
            for r in results:
                writer.writerow(
                    {
                        "kidney_key": r["kidney_key"],
                        "status": r["status"],
                        "ct_sdm_path": r.get("ct_sdm_path") or "",
                        "us_sdm_path": r.get("us_sdm_path") or "",
                        "errors": "; ".join(r.get("errors", [])),
                        "warnings": "; ".join(r.get("warnings", [])),
                    }
                )
        summary_path = args.out_dir / "validation_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

    logger.info(
        "Done: %d processed, %d skipped, %d failed, %d missing, %d empty, %d metadata issues",
        processed,
        skipped,
        failed,
        missing,
        empty_masks,
        metadata_issues,
    )
    logger.info("Report: %s", report_path)

    if args.fail_on_mismatch and metadata_issues > 0:
        return 1
    if failed > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
