#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import affine_transform

from pca_icp.dataset import build_dataset_index
from pca_icp.pipeline import register_nii_segmentations
from pca_icp.types import RegistrationConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _matrix_to_list(mat: np.ndarray) -> list[list[float]]:
    return [[float(x) for x in row] for row in mat]


def _to_jsonable(value):
    """
    Convert NumPy-heavy nested structures into JSON-serializable Python types.
    """
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _load_binary_mask(path: str, label: int) -> tuple[np.ndarray, np.ndarray]:
    img = nib.load(path)
    data = np.asanyarray(img.dataobj)
    if data.ndim != 3:
        raise ValueError(f"Expected 3D mask for Dice, got shape {data.shape} from {path}")
    mask = (data == label).astype(np.uint8)
    return mask, img.affine


def _dice_coefficient(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a > 0
    b = mask_b > 0
    a_sum = int(a.sum())
    b_sum = int(b.sum())
    if a_sum == 0 and b_sum == 0:
        return 1.0
    if a_sum == 0 or b_sum == 0:
        return 0.0
    intersection = int(np.logical_and(a, b).sum())
    return float((2.0 * intersection) / (a_sum + b_sum))


def _resample_source_mask_to_target_grid(
    source_mask: np.ndarray,
    source_affine: np.ndarray,
    target_shape: tuple[int, int, int],
    target_affine: np.ndarray,
    source_to_target_world: np.ndarray,
) -> np.ndarray:
    """
    Resample source mask into target voxel grid using nearest-neighbor interpolation.
    Coordinate chain:
      i_src -> world_src -> world_tgt -> i_tgt
    where source_to_target_world maps world_src to world_tgt.
    """
    src_to_tgt_idx = np.linalg.inv(target_affine) @ source_to_target_world @ source_affine
    tgt_to_src_idx = np.linalg.inv(src_to_tgt_idx)

    matrix = tgt_to_src_idx[:3, :3]
    offset = tgt_to_src_idx[:3, 3]
    resampled = affine_transform(
        source_mask.astype(np.float32),
        matrix=matrix,
        offset=offset,
        output_shape=target_shape,
        order=0,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    return (resampled > 0.5).astype(np.uint8)


def _save_tfm_for_slicer(ras_matrix: np.ndarray, out_path: Path) -> None:
    """
    Save transform as ITK .tfm in LPS coordinate convention for Slicer.
    """
    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise RuntimeError(
            "SimpleITK is required to write .tfm transforms. Install with: pip install SimpleITK"
        ) from exc

    ras_to_lps = np.diag([-1.0, -1.0, 1.0, 1.0])
    lps_matrix = ras_to_lps @ ras_matrix @ ras_to_lps

    tx = sitk.AffineTransform(3)
    tx.SetMatrix(lps_matrix[:3, :3].reshape(-1).tolist())
    tx.SetTranslation(lps_matrix[:3, 3].tolist())
    sitk.WriteTransform(tx, str(out_path))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run PCA + ICP registration on CT/US segmentation pairs.")
    parser.add_argument("--base_dir", type=Path, required=True, help="Dataset root with CT_masks and US_masks.")
    parser.add_argument("--out_dir", type=Path, required=True, help="Output directory for per-case JSON reports.")
    parser.add_argument("--source_label", type=int, default=1)
    parser.add_argument("--target_label", type=int, default=1)
    parser.add_argument("--smoothing_iterations", type=int, default=10)
    parser.add_argument("--decimation_reduction", type=float, default=0.0)
    parser.add_argument("--pca_unstable_threshold", type=float, default=1.10)
    parser.add_argument("--icp_mode", type=str, default="rigid", choices=["rigid", "similarity", "affine"])
    parser.add_argument("--icp_max_iterations", type=int, default=100)
    parser.add_argument("--icp_max_landmarks", type=int, default=1000)
    parser.add_argument("--icp_max_mean_distance", type=float, default=1e-3)
    parser.add_argument("--multistart_top_k", type=int, default=2)
    parser.add_argument(
        "--save_tfm",
        action="store_true",
        help="Save final US->CT transform as .tfm for Slicer.",
    )
    parser.add_argument(
        "--transform_dir",
        type=Path,
        default=None,
        help="Directory for .tfm outputs (default: out_dir).",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    transform_dir = args.transform_dir or args.out_dir
    if args.save_tfm:
        transform_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset_index(str(args.base_dir))
    if not dataset:
        logger.error("No valid CT/US mask pairs found in %s", args.base_dir)
        return 1

    config = RegistrationConfig(
        source_label=args.source_label,
        target_label=args.target_label,
        smoothing_iterations=args.smoothing_iterations,
        decimation_reduction=args.decimation_reduction,
        pca_unstable_threshold=args.pca_unstable_threshold,
        icp_mode=args.icp_mode,
        icp_max_iterations=args.icp_max_iterations,
        icp_max_landmarks=args.icp_max_landmarks,
        icp_max_mean_distance=args.icp_max_mean_distance,
        multistart_top_k=args.multistart_top_k,
    )

    failures = 0
    for case_id in sorted(dataset.keys()):
        pair = dataset[case_id]
        try:
            result = register_nii_segmentations(
                source_nii_path=pair["us_mask"],
                target_nii_path=pair["ct_mask"],
                config=config,
            )

            source_mask, source_affine = _load_binary_mask(pair["us_mask"], args.source_label)
            target_mask, target_affine = _load_binary_mask(pair["ct_mask"], args.target_label)
            moved_source_mask = _resample_source_mask_to_target_grid(
                source_mask=source_mask,
                source_affine=source_affine,
                target_shape=target_mask.shape,
                target_affine=target_affine,
                source_to_target_world=result.final_matrix,
            )
            dice_score = _dice_coefficient(moved_source_mask, target_mask)

            tfm_path = None
            if args.save_tfm:
                tfm_path = transform_dir / f"{case_id}_US_to_CT_pca_icp.tfm"
                _save_tfm_for_slicer(result.final_matrix, tfm_path)

            payload = {
                "case_id": case_id,
                "best_candidate_name": result.best_candidate_name,
                "best_candidate_score": float(result.best_candidate_score),
                "dice_score": float(dice_score),
                "pca_matrix": _matrix_to_list(result.pca_matrix),
                "icp_matrix": _matrix_to_list(result.icp_matrix),
                "final_matrix": _matrix_to_list(result.final_matrix),
                "final_transform_tfm_path": str(tfm_path) if tfm_path is not None else None,
                "candidate_scores": [
                    {
                        "candidate_name": s.candidate_name,
                        "symmetric_mean_distance": float(s.symmetric_mean_distance),
                        "hausdorff95": float(s.hausdorff95),
                    }
                    for s in result.candidate_scores
                ],
                "diagnostics": _to_jsonable(result.diagnostics),
            }
            out_path = args.out_dir / f"{case_id}_pca_icp_report.json"
            out_path.write_text(json.dumps(_to_jsonable(payload), indent=2), encoding="utf-8")
            logger.info(
                "Completed %s -> %s (Dice=%.4f)",
                case_id,
                out_path.name,
                dice_score,
            )
        except Exception as exc:
            failures += 1
            logger.exception("Failed case %s: %s", case_id, exc)

    if failures > 0:
        logger.error("%d case(s) failed", failures)
        return 1
    logger.info("All %d case(s) completed successfully", len(dataset))
    return 0


if __name__ == "__main__":
    sys.exit(main())
