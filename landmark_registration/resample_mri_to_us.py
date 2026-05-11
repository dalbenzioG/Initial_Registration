#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from landmark_registration.transform import matrix4x4_to_sitk_affine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be object: {path}")
    return payload


def _resolve_report_paths(summary_path: Path, case_id: str | None) -> list[Path]:
    summary = _load_json(summary_path)
    cases = summary.get("cases")
    if not isinstance(cases, list):
        raise ValueError(f"Summary has no valid 'cases' array: {summary_path}")

    out: list[Path] = []
    for case in cases:
        if not isinstance(case, dict):
            continue
        if case.get("status") != "success":
            continue
        if case_id is not None and case.get("case_id") != case_id:
            continue
        report_path = case.get("report_path")
        if not isinstance(report_path, str) or not report_path.strip():
            logger.warning("Skipping summary entry with missing report_path: %s", case)
            continue
        out.append(Path(report_path))

    if case_id is not None and not out:
        raise ValueError(f"Case '{case_id}' not found among successful entries in {summary_path}")
    return out


def _resample_moving_to_fixed(
    sitk,
    moving,
    fixed,
    tx_moving_to_fixed,
    *,
    interpolation: str,
    default_value: float,
):
    interpolator = sitk.sitkLinear if interpolation == "linear" else sitk.sitkNearestNeighbor
    tx_fixed_to_moving = tx_moving_to_fixed.GetInverse()
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetTransform(tx_fixed_to_moving)
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(float(default_value))
    return resampler.Execute(moving)


def _case_output_name(case_id: str, model: str, suffix: str) -> str:
    return f"{case_id}_MRI_in_US_landmark_{model}{suffix}"


def _run_case_from_report(
    sitk,
    report_path: Path,
    out_dir: Path,
    *,
    interpolation: str,
    default_value: float,
    suffix: str,
) -> Path:
    report = _load_json(report_path)
    required = ("case_id", "moving_image_path", "fixed_image_path", "transform_matrix_4x4", "model")
    missing = [k for k in required if k not in report]
    if missing:
        raise ValueError(f"Report missing required fields {missing}: {report_path}")

    case_id = str(report["case_id"]).strip()
    if not case_id:
        raise ValueError(f"Invalid empty case_id in report: {report_path}")

    moving_path = Path(str(report["moving_image_path"]))
    fixed_path = Path(str(report["fixed_image_path"]))
    if not moving_path.is_file():
        raise FileNotFoundError(f"Moving MRI not found: {moving_path}")
    if not fixed_path.is_file():
        raise FileNotFoundError(f"Fixed US not found: {fixed_path}")

    transform_4x4 = report["transform_matrix_4x4"]
    tx = matrix4x4_to_sitk_affine(transform_4x4)

    logger.info("Case %s: loading MRI moving image %s", case_id, moving_path)
    moving = sitk.ReadImage(str(moving_path))
    logger.info("Case %s: loading US fixed image %s", case_id, fixed_path)
    fixed = sitk.ReadImage(str(fixed_path))

    resampled = _resample_moving_to_fixed(
        sitk,
        moving,
        fixed,
        tx,
        interpolation=interpolation,
        default_value=default_value,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    model = str(report.get("model", "rigid"))
    out_path = out_dir / _case_output_name(case_id, model, suffix)
    use_compression = suffix.endswith(".gz")
    sitk.WriteImage(resampled, str(out_path), use_compression)
    logger.info("Case %s: wrote %s", case_id, out_path)
    return out_path


def main() -> int:
    try:
        import SimpleITK as sitk
    except ImportError:
        logger.exception("SimpleITK is required. Install with `pip install SimpleITK`.")
        return 1

    parser = argparse.ArgumentParser(
        description="Resample MRI into US space using landmark_registration report transforms."
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--summary_json",
        type=Path,
        help="Path to landmark_registration_summary.json; all successful case reports are processed.",
    )
    src.add_argument(
        "--report_json",
        type=Path,
        help="Path to a single <case_id>_landmark_report.json file.",
    )
    parser.add_argument(
        "--case_id",
        type=str,
        default=None,
        help="Optional case filter for --summary_json mode.",
    )
    parser.add_argument("--out_dir", type=Path, required=True, help="Output directory for resampled MRI files.")
    parser.add_argument(
        "--interpolation",
        type=str,
        choices=("linear", "nearest"),
        default="linear",
        help="Interpolation mode (linear default; nearest for labels).",
    )
    parser.add_argument(
        "--default_value",
        type=float,
        default=0.0,
        help="Value used when fixed-grid voxels sample outside moving MRI domain.",
    )
    parser.add_argument(
        "--out_suffix",
        type=str,
        choices=(".nii", ".nii.gz"),
        default=".nii.gz",
        help="Output NIfTI suffix.",
    )
    args = parser.parse_args()

    if args.summary_json is not None:
        if not args.summary_json.is_file():
            logger.error("Not a file: %s", args.summary_json)
            return 1
        report_paths = _resolve_report_paths(args.summary_json, args.case_id)
        if not report_paths:
            logger.error("No successful report paths found in %s", args.summary_json)
            return 1
    else:
        if args.case_id is not None:
            logger.error("--case_id is only supported with --summary_json")
            return 1
        if args.report_json is None or not args.report_json.is_file():
            logger.error("Not a file: %s", args.report_json)
            return 1
        report_paths = [args.report_json]

    failures = 0
    for report_path in report_paths:
        try:
            _run_case_from_report(
                sitk,
                report_path,
                args.out_dir,
                interpolation=args.interpolation,
                default_value=args.default_value,
                suffix=args.out_suffix,
            )
        except Exception as exc:
            failures += 1
            logger.exception("Failed report %s: %s", report_path, exc)

    if failures:
        logger.error("Resampling finished with %d failure(s).", failures)
        return 1
    logger.info("Resampling completed for %d case(s).", len(report_paths))
    return 0


if __name__ == "__main__":
    sys.exit(main())
