#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

from landmark_registration.io import ManifestCase, load_manifest_cases, read_mni_tag_file
from landmark_registration.metrics import compute_lc2_metric, compute_landmark_errors
from landmark_registration.transform import (
    VALID_MODELS,
    compute_landmarks_transform,
    save_matrix_tfm,
    save_matrix_txt,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _matrix_to_list(mat: np.ndarray) -> list[list[float]]:
    return [[float(x) for x in row] for row in mat]


def _compute_lc2_block(
    case: ManifestCase,
    transform_4x4: np.ndarray,
) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []

    moving_path = case.moving_image_path
    fixed_path = case.fixed_image_path
    if moving_path is None or fixed_path is None:
        return {
            "status": "skipped_missing_paths",
            "before": None,
            "after": None,
            "delta": None,
        }, warnings

    if not moving_path.exists() or not fixed_path.exists():
        warnings.append(
            f"LC2 skipped because moving/fixed images do not exist: "
            f"{moving_path} | {fixed_path}"
        )
        return {
            "status": "skipped_missing_files",
            "before": None,
            "after": None,
            "delta": None,
        }, warnings

    try:
        # Before uses native image relationship; this is valid when geometry is already comparable.
        # If the images are not on matching grid/shape this can raise and we gracefully skip.
        lc2_before = compute_lc2_metric(
            moving_image_path=moving_path,
            fixed_image_path=fixed_path,
            moving_to_fixed_world=None,
        )
        lc2_after = compute_lc2_metric(
            moving_image_path=moving_path,
            fixed_image_path=fixed_path,
            moving_to_fixed_world=transform_4x4,
        )
    except Exception as exc:
        warnings.append(f"LC2 skipped due to computation error: {exc}")
        return {
            "status": "skipped_error",
            "before": None,
            "after": None,
            "delta": None,
        }, warnings

    return {
        "status": "computed",
        "before": float(lc2_before),
        "after": float(lc2_after),
        "delta": float(lc2_after - lc2_before),
    }, warnings


def _build_case_payload(
    *,
    case: ManifestCase,
    moving_modality: str,
    fixed_modality: str,
    model: str,
    n_landmarks: int,
    transform_4x4: np.ndarray,
    matrix_txt_path: Path,
    matrix_tfm_path: Path,
    tre_before: dict[str, Any],
    tre_after: dict[str, Any],
    lc2: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    return {
        "status": "success",
        "case_id": case.case_id,
        "moving_modality": moving_modality,
        "fixed_modality": fixed_modality,
        "transform_direction": f"{moving_modality}_to_{fixed_modality}",
        "model": model,
        "tag_path": str(case.tag_path),
        "moving_image_path": str(case.moving_image_path) if case.moving_image_path is not None else None,
        "fixed_image_path": str(case.fixed_image_path) if case.fixed_image_path is not None else None,
        "n_landmarks": int(n_landmarks),
        "transform_matrix_txt_path": str(matrix_txt_path),
        "transform_matrix_tfm_path": str(matrix_tfm_path),
        "transform_matrix_4x4": _matrix_to_list(transform_4x4),
        "tre_mm": {
            "before": tre_before,
            "after": tre_after,
            "delta_rmse": float(tre_after["rmse"] - tre_before["rmse"]),
            "delta_mean": float(tre_after["mean"] - tre_before["mean"]),
        },
        "lc2": lc2,
        "warnings": warnings,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch landmark-based registration from MNI .tag files using JSON manifest."
    )
    parser.add_argument("--manifest", type=Path, required=True, help="Path to JSON manifest.")
    parser.add_argument("--out_dir", type=Path, required=True, help="Output directory for matrices/reports.")
    parser.add_argument(
        "--model",
        type=str,
        default="rigid",
        choices=list(VALID_MODELS),
        help="Landmark transform model.",
    )
    parser.add_argument("--moving_modality", type=str, default="MRI")
    parser.add_argument("--fixed_modality", type=str, default="US")

    parser.add_argument("--manifest_cases_key", type=str, default="cases")
    parser.add_argument("--manifest_case_id_key", type=str, default="case_id")
    parser.add_argument("--manifest_tag_key", type=str, default="tag_path")
    parser.add_argument("--manifest_moving_img_key", type=str, default="moving_image_path")
    parser.add_argument("--manifest_fixed_img_key", type=str, default="fixed_image_path")

    parser.add_argument(
        "--save_per_point_errors",
        action="store_true",
        help="Persist per-point TRE arrays in per-case JSON reports.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    matrix_dir = args.out_dir / "matrices"
    report_dir = args.out_dir / "reports"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    try:
        cases = load_manifest_cases(
            args.manifest,
            cases_key=args.manifest_cases_key,
            case_id_key=args.manifest_case_id_key,
            tag_key=args.manifest_tag_key,
            moving_img_key=args.manifest_moving_img_key,
            fixed_img_key=args.manifest_fixed_img_key,
        )
    except Exception as exc:
        logger.exception("Failed to parse manifest: %s", exc)
        return 1

    failures = 0
    case_summaries: list[dict[str, Any]] = []

    for case in cases:
        logger.info("Processing case %s", case.case_id)
        case_warnings: list[str] = []
        try:
            moving_pts, fixed_pts = read_mni_tag_file(case.tag_path)
            transform_4x4 = compute_landmarks_transform(
                moving_points=moving_pts,
                fixed_points=fixed_pts,
                model=args.model,
            )

            tre_before = compute_landmark_errors(moving_pts, fixed_pts, transform_4x4=None)
            tre_after = compute_landmark_errors(moving_pts, fixed_pts, transform_4x4=transform_4x4)
            if not args.save_per_point_errors:
                tre_before = {k: v for k, v in tre_before.items() if k != "errors"}
                tre_after = {k: v for k, v in tre_after.items() if k != "errors"}
                n_landmarks = moving_pts.shape[0]
                tre_before["n_landmarks"] = int(n_landmarks)
                tre_after["n_landmarks"] = int(n_landmarks)

            lc2_payload, lc2_warnings = _compute_lc2_block(case, transform_4x4)
            case_warnings.extend(lc2_warnings)

            matrix_txt_path = matrix_dir / (
                f"{case.case_id}_{args.moving_modality}_to_{args.fixed_modality}_landmark_{args.model}.txt"
            )
            matrix_tfm_path = matrix_txt_path.with_suffix(".tfm")
            save_matrix_txt(transform_4x4, matrix_txt_path)
            save_matrix_tfm(transform_4x4, matrix_tfm_path)

            payload = _build_case_payload(
                case=case,
                moving_modality=args.moving_modality,
                fixed_modality=args.fixed_modality,
                model=args.model,
                n_landmarks=moving_pts.shape[0],
                transform_4x4=transform_4x4,
                matrix_txt_path=matrix_txt_path,
                matrix_tfm_path=matrix_tfm_path,
                tre_before=tre_before,
                tre_after=tre_after,
                lc2=lc2_payload,
                warnings=case_warnings,
            )
            out_report = report_dir / f"{case.case_id}_landmark_report.json"
            out_report.write_text(json.dumps(_to_jsonable(payload), indent=2), encoding="utf-8")

            case_summaries.append(
                {
                    "case_id": case.case_id,
                    "status": "success",
                    "report_path": str(out_report),
                    "matrix_path": str(matrix_txt_path),
                    "matrix_tfm_path": str(matrix_tfm_path),
                    "tre_rmse_before": float(payload["tre_mm"]["before"]["rmse"]),
                    "tre_rmse_after": float(payload["tre_mm"]["after"]["rmse"]),
                    "lc2_status": payload["lc2"]["status"],
                    "lc2_before": payload["lc2"]["before"],
                    "lc2_after": payload["lc2"]["after"],
                }
            )
            logger.info(
                "Completed %s (RMSE %.4f -> %.4f)",
                case.case_id,
                payload["tre_mm"]["before"]["rmse"],
                payload["tre_mm"]["after"]["rmse"],
            )
        except Exception as exc:
            failures += 1
            logger.exception("Failed case %s: %s", case.case_id, exc)
            case_summaries.append(
                {
                    "case_id": case.case_id,
                    "status": "failed",
                    "error": str(exc),
                    "tag_path": str(case.tag_path),
                }
            )

    summary = {
        "manifest_path": str(args.manifest),
        "out_dir": str(args.out_dir),
        "model": args.model,
        "moving_modality": args.moving_modality,
        "fixed_modality": args.fixed_modality,
        "total_cases": len(cases),
        "successful_cases": len(cases) - failures,
        "failed_cases": failures,
        "cases": case_summaries,
    }
    summary_path = args.out_dir / "landmark_registration_summary.json"
    summary_path.write_text(json.dumps(_to_jsonable(summary), indent=2), encoding="utf-8")

    if failures > 0:
        logger.error("%d case(s) failed. Summary: %s", failures, summary_path)
        return 1

    logger.info("All %d case(s) completed. Summary: %s", len(cases), summary_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
