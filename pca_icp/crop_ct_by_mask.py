#!/usr/bin/env python3
"""
Mask-aware in-plane cropping for CT images and CT masks.

This script crops X/Y to a fixed size (default 512x512) while keeping all Z slices.
For each case, crop placement is centered on the CT mask bounding box so kidney
voxels stay inside the output field of view. When the source image is smaller than
the requested crop size, the script pads (CT: configurable value, mask: 0).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from pca_icp.dataset import extract_case_id

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _nifti_stem(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[: -len(".nii.gz")]
    if name.endswith(".nii"):
        return name[: -len(".nii")]
    return path.stem


def _collect_single_file_per_case(directory: Path, role: str) -> dict[str, Path]:
    if not directory.is_dir():
        logger.error("Not a directory (%s): %s", role, directory)
        return {}

    buckets: dict[str, list[Path]] = {}
    for p in sorted(directory.iterdir()):
        if not p.is_file():
            continue
        case_id = extract_case_id(p.name)
        if case_id is None:
            continue
        buckets.setdefault(case_id, []).append(p)

    out: dict[str, Path] = {}
    for case_id, paths in buckets.items():
        if len(paths) > 1:
            logger.warning(
                "Skipping %s for %s: multiple files for case %s: %s",
                role,
                directory,
                case_id,
                [x.name for x in paths],
            )
            continue
        out[case_id] = paths[0]
    return out


def _write_nifti_checked(sitk, image, path: Path, *, use_compression: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = sitk.ImageFileWriter()
    writer.SetFileName(str(path))
    writer.SetUseCompression(use_compression)
    writer.Execute(image)
    if not path.is_file():
        raise RuntimeError(f"NIfTI write did not create file: {path}")
    if path.stat().st_size == 0:
        raise RuntimeError(f"NIfTI write produced 0-byte file: {path}")


def _mask_bbox_xy(mask_zyx: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask_zyx > 0)
    if coords.size == 0:
        raise ValueError("Mask has no positive voxels.")
    y_min = int(coords[:, 1].min())
    y_max = int(coords[:, 1].max())
    x_min = int(coords[:, 2].min())
    x_max = int(coords[:, 2].max())
    return x_min, x_max, y_min, y_max


def _choose_crop_start(
    size_x: int,
    size_y: int,
    target_x: int,
    target_y: int,
    bbox_xy: tuple[int, int, int, int],
    margin_x: int,
    margin_y: int,
) -> tuple[int, int]:
    x_min, x_max, y_min, y_max = bbox_xy
    roi_x_min = x_min - margin_x
    roi_x_max = x_max + margin_x
    roi_y_min = y_min - margin_y
    roi_y_max = y_max + margin_y

    roi_w = roi_x_max - roi_x_min + 1
    roi_h = roi_y_max - roi_y_min + 1
    if roi_w > target_x:
        raise ValueError(f"Mask ROI width ({roi_w}) exceeds target crop width ({target_x}).")
    if roi_h > target_y:
        raise ValueError(f"Mask ROI height ({roi_h}) exceeds target crop height ({target_y}).")

    center_x = 0.5 * (roi_x_min + roi_x_max)
    center_y = 0.5 * (roi_y_min + roi_y_max)

    src_x0 = int(np.floor(center_x - (target_x / 2.0)))
    src_y0 = int(np.floor(center_y - (target_y / 2.0)))

    if size_x >= target_x:
        src_x0 = max(0, min(src_x0, size_x - target_x))
    if size_y >= target_y:
        src_y0 = max(0, min(src_y0, size_y - target_y))

    return src_x0, src_y0


def _crop_or_pad_zyx(
    array_zyx: np.ndarray,
    src_x0: int,
    src_y0: int,
    target_x: int,
    target_y: int,
    pad_value: float,
) -> np.ndarray:
    if array_zyx.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {array_zyx.shape}")

    size_z, size_y, size_x = array_zyx.shape
    out = np.full(
        (size_z, target_y, target_x),
        fill_value=pad_value,
        dtype=array_zyx.dtype,
    )

    x_in0 = max(0, src_x0)
    x_in1 = min(size_x, src_x0 + target_x)
    y_in0 = max(0, src_y0)
    y_in1 = min(size_y, src_y0 + target_y)
    if x_in0 >= x_in1 or y_in0 >= y_in1:
        return out

    x_out0 = max(0, -src_x0)
    y_out0 = max(0, -src_y0)
    x_out1 = x_out0 + (x_in1 - x_in0)
    y_out1 = y_out0 + (y_in1 - y_in0)

    out[:, y_out0:y_out1, x_out0:x_out1] = array_zyx[:, y_in0:y_in1, x_in0:x_in1]
    return out


def _crop_pair(
    sitk,
    ct_img,
    ct_mask,
    target_x: int,
    target_y: int,
    margin_x: int,
    margin_y: int,
    ct_pad_value: float,
):
    ct_arr = sitk.GetArrayFromImage(ct_img)
    mask_arr = sitk.GetArrayFromImage(ct_mask)

    if ct_arr.shape != mask_arr.shape:
        raise ValueError(
            f"CT image and CT mask shapes differ: image={ct_arr.shape}, mask={mask_arr.shape}"
        )

    _, size_y, size_x = ct_arr.shape
    bbox_xy = _mask_bbox_xy(mask_arr)
    src_x0, src_y0 = _choose_crop_start(
        size_x=size_x,
        size_y=size_y,
        target_x=target_x,
        target_y=target_y,
        bbox_xy=bbox_xy,
        margin_x=margin_x,
        margin_y=margin_y,
    )

    ct_crop = _crop_or_pad_zyx(
        ct_arr,
        src_x0=src_x0,
        src_y0=src_y0,
        target_x=target_x,
        target_y=target_y,
        pad_value=ct_pad_value,
    )
    mask_crop = _crop_or_pad_zyx(
        mask_arr,
        src_x0=src_x0,
        src_y0=src_y0,
        target_x=target_x,
        target_y=target_y,
        pad_value=0,
    )

    original_mask_voxels = int((mask_arr > 0).sum())
    cropped_mask_voxels = int((mask_crop > 0).sum())
    if cropped_mask_voxels != original_mask_voxels:
        raise RuntimeError(
            f"Cropping would lose mask voxels ({cropped_mask_voxels}/{original_mask_voxels} kept)."
        )

    ct_out = sitk.GetImageFromArray(ct_crop)
    mask_out = sitk.GetImageFromArray(mask_crop.astype(mask_arr.dtype))
    ct_out.SetSpacing(ct_img.GetSpacing())
    ct_out.SetDirection(ct_img.GetDirection())
    mask_out.SetSpacing(ct_mask.GetSpacing())
    mask_out.SetDirection(ct_mask.GetDirection())

    # New origin is the physical point of the source index mapped to output (0, 0, 0).
    ct_origin = ct_img.TransformContinuousIndexToPhysicalPoint((float(src_x0), float(src_y0), 0.0))
    mask_origin = ct_mask.TransformContinuousIndexToPhysicalPoint((float(src_x0), float(src_y0), 0.0))
    ct_out.SetOrigin(ct_origin)
    mask_out.SetOrigin(mask_origin)

    info = {
        "source_shape_zyx": [int(v) for v in ct_arr.shape],
        "target_shape_zyx": [int(v) for v in ct_crop.shape],
        "mask_bbox_xy": {
            "x_min": int(bbox_xy[0]),
            "x_max": int(bbox_xy[1]),
            "y_min": int(bbox_xy[2]),
            "y_max": int(bbox_xy[3]),
        },
        "crop_start_xy": {"x": int(src_x0), "y": int(src_y0)},
        "mask_voxels_original": original_mask_voxels,
        "mask_voxels_cropped": cropped_mask_voxels,
    }
    return ct_out, mask_out, info


def main() -> int:
    import SimpleITK as sitk

    parser = argparse.ArgumentParser(
        description="Crop CT_images and CT_masks to fixed XY size using mask-aware placement."
    )
    parser.add_argument(
        "--batch_root",
        type=Path,
        required=True,
        help="Dataset root containing CT_images/ and CT_masks/.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        required=True,
        help="Output root for cropped_CT_images/, cropped_CT_masks/, and crop_reports/.",
    )
    parser.add_argument(
        "--ct_images_subdir",
        type=str,
        default="CT_images",
        help="Input CT image subfolder under --batch_root (default: CT_images).",
    )
    parser.add_argument(
        "--ct_masks_subdir",
        type=str,
        default="CT_masks",
        help="Input CT mask subfolder under --batch_root (default: CT_masks).",
    )
    parser.add_argument(
        "--target_xy",
        type=int,
        nargs=2,
        default=(512, 512),
        metavar=("X", "Y"),
        help="Requested output in-plane size in voxels (default: 512 512).",
    )
    parser.add_argument(
        "--bbox_margin_xy",
        type=int,
        nargs=2,
        default=(12, 12),
        metavar=("MX", "MY"),
        help="Extra margin around mask bounding box before centering crop (default: 12 12).",
    )
    parser.add_argument(
        "--ct_pad_value",
        type=float,
        default=-1000.0,
        help="Pad value used for CT outside original FOV (default: -1000).",
    )
    parser.add_argument(
        "--nifti_suffix",
        type=str,
        default=".nii.gz",
        choices=(".nii", ".nii.gz"),
        help="Output file suffix (default: .nii.gz).",
    )
    parser.add_argument(
        "--nifti_gzip",
        action="store_true",
        help="Enable gzip compression when writing .nii.gz outputs.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Stop on first case failure; otherwise continue with remaining cases.",
    )
    args = parser.parse_args()

    target_x, target_y = int(args.target_xy[0]), int(args.target_xy[1])
    margin_x, margin_y = int(args.bbox_margin_xy[0]), int(args.bbox_margin_xy[1])
    if target_x <= 0 or target_y <= 0:
        logger.error("--target_xy values must be positive.")
        return 1
    if margin_x < 0 or margin_y < 0:
        logger.error("--bbox_margin_xy values must be >= 0.")
        return 1

    ct_img_dir = args.batch_root / args.ct_images_subdir
    ct_mask_dir = args.batch_root / args.ct_masks_subdir
    img_by_case = _collect_single_file_per_case(ct_img_dir, "CT_images")
    mask_by_case = _collect_single_file_per_case(ct_mask_dir, "CT_masks")

    all_cases = sorted(img_by_case.keys() | mask_by_case.keys())
    case_ids = sorted(img_by_case.keys() & mask_by_case.keys())
    for case_id in all_cases:
        missing = []
        if case_id not in img_by_case:
            missing.append(args.ct_images_subdir)
        if case_id not in mask_by_case:
            missing.append(args.ct_masks_subdir)
        if missing:
            logger.warning("Incomplete case %s under %s (missing %s)", case_id, args.batch_root, ", ".join(missing))

    if not case_ids:
        logger.error("No paired CT image/mask cases found in %s", args.batch_root)
        return 1

    out_img_dir = args.out_dir / "cropped_CT_images"
    out_mask_dir = args.out_dir / "cropped_CT_masks"
    out_report_dir = args.out_dir / "crop_reports"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)
    out_report_dir.mkdir(parents=True, exist_ok=True)

    use_compression = bool(args.nifti_gzip and args.nifti_suffix == ".nii.gz")
    if args.nifti_suffix == ".nii.gz" and not args.nifti_gzip:
        logger.warning(
            "Using suffix .nii.gz without gzip compression (--nifti_gzip not set). "
            "File contents are uncompressed NIfTI."
        )

    failures = 0
    successes = 0
    for case_id in case_ids:
        ct_path = img_by_case[case_id]
        mask_path = mask_by_case[case_id]
        try:
            logger.info("Case %s: loading %s and %s", case_id, ct_path.name, mask_path.name)
            ct_img = sitk.ReadImage(str(ct_path))
            ct_mask = sitk.ReadImage(str(mask_path))

            ct_crop, mask_crop, info = _crop_pair(
                sitk=sitk,
                ct_img=ct_img,
                ct_mask=ct_mask,
                target_x=target_x,
                target_y=target_y,
                margin_x=margin_x,
                margin_y=margin_y,
                ct_pad_value=args.ct_pad_value,
            )

            out_ct = out_img_dir / f"{_nifti_stem(ct_path)}_crop{target_x}x{target_y}{args.nifti_suffix}"
            out_mask = out_mask_dir / f"{_nifti_stem(mask_path)}_crop{target_x}x{target_y}{args.nifti_suffix}"
            report_path = out_report_dir / f"{case_id}_crop_report.json"

            _write_nifti_checked(sitk, ct_crop, out_ct, use_compression=use_compression)
            _write_nifti_checked(sitk, mask_crop, out_mask, use_compression=use_compression)
            report_payload = {
                "case_id": case_id,
                "ct_input": str(ct_path),
                "ct_mask_input": str(mask_path),
                "ct_output": str(out_ct),
                "ct_mask_output": str(out_mask),
                "target_xy": [target_x, target_y],
                "bbox_margin_xy": [margin_x, margin_y],
                "ct_pad_value": float(args.ct_pad_value),
                "details": info,
            }
            report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
            successes += 1
            logger.info("Case %s: wrote %s and %s", case_id, out_ct.name, out_mask.name)
        except Exception as exc:
            failures += 1
            logger.exception("Case %s failed: %s", case_id, exc)
            if args.strict:
                return 1

    logger.info("Finished cropping: %d succeeded, %d failed", successes, failures)
    return 0 if successes > 0 and (failures == 0 or not args.strict) else 1


if __name__ == "__main__":
    sys.exit(main())
