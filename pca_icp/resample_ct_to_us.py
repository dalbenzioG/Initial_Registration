#!/usr/bin/env python3
"""
Resample a CT volume onto the US reference grid using an existing CT->US transform.

Transform sources:
  - .tfm written by run_pca_icp_registration --save_tfm (ITK/LPS)
  - resample_matrix from a *_pca_icp_report.json (RAS 4x4, same convention as the runner)

Batch mode (--batch_root): TRUSTED-style layout (CT_images, CT_masks, US_masks, init_transf)
resamples CT volumes and CT masks per case_id.
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
    """Filename without .nii.gz / .nii suffix."""
    name = path.name
    if name.endswith(".nii.gz"):
        return name[: -len(".nii.gz")]
    if name.endswith(".nii"):
        return name[: -len(".nii")]
    return path.stem


def _write_nifti_checked(sitk, image, path: Path, *, use_compression: bool) -> None:
    """
    Write NIfTI via ImageFileWriter and verify the file is non-empty.

    ITK may print ** ERROR: NWAD: wrote only 0 of N bytes ... to stderr on failure
    (disk full, quota, flaky NFS) without always raising; empty files must be treated as errors.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = sitk.ImageFileWriter()
    writer.SetFileName(str(path))
    writer.SetUseCompression(use_compression)
    writer.Execute(image)
    if not path.is_file():
        raise RuntimeError(f"NIfTI write did not create file: {path}")
    size = path.stat().st_size
    if size == 0:
        raise RuntimeError(
            f"NIfTI write produced 0-byte file: {path}. "
            "Check disk space or inode quota; on network filesystems prefer uncompressed "
            "batch outputs (--batch_nifti_suffix .nii, default)."
        )


def _maybe_float32_ct(sitk, image):
    """Linear resampling can promote to Float64; halve I/O size for large volumes."""
    if image.GetPixelID() == sitk.sitkFloat64:
        return sitk.Cast(image, sitk.sitkFloat32)
    return image


def _resample_moving_to_fixed(
    moving,
    fixed,
    tx_moving_to_fixed,
    interpolator: int,
    default_value: float,
):
    """
    Resample moving image into fixed image space.

    tx_moving_to_fixed: maps moving physical coordinates -> fixed physical coordinates.
    ResampleImageFilter uses the inverse to sample moving voxels at each fixed voxel.
    """
    import SimpleITK as sitk

    tx_fixed_to_moving = tx_moving_to_fixed.GetInverse()
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetTransform(tx_fixed_to_moving)
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(float(default_value))
    return resampler.Execute(moving)


def _affine_from_ras_matrix(ras_4x4: np.ndarray):
    """Build SimpleITK AffineTransform matching _save_tfm_for_slicer (RAS -> LPS for ITK)."""
    import SimpleITK as sitk

    ras_to_lps = np.diag([-1.0, -1.0, 1.0, 1.0])
    lps_matrix = ras_to_lps @ ras_4x4 @ ras_to_lps

    tx = sitk.AffineTransform(3)
    tx.SetMatrix(lps_matrix[:3, :3].reshape(-1).tolist())
    tx.SetTranslation(lps_matrix[:3, 3].tolist())
    return tx


def _load_transform_from_report_json(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    key = "resample_matrix"
    if key not in data:
        raise KeyError(f"{path} has no '{key}' field")
    mat = np.asarray(data[key], dtype=np.float64)
    if mat.shape != (4, 4):
        raise ValueError(f"{key} must be 4x4, got {mat.shape}")
    name = data.get("resample_transform_name", "")
    if name and name != "CT_to_US":
        logger.warning(
            "Expected resample_transform_name 'CT_to_US' for CT->US resampling; got %r",
            name,
        )
    return _affine_from_ras_matrix(mat)


def _collect_single_file_per_case(directory: Path, role: str) -> dict[str, Path]:
    """
    Map case_id -> file path for files whose names start with case_id pattern.
    Skips filenames that do not match extract_case_id.
    If multiple files share a case_id, that case_id is omitted and a warning is logged.
    """
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


def _run_single_case(
    sitk,
    ct: Path,
    us_reference: Path,
    out: Path,
    tfm: Path | None,
    report_json: Path | None,
    default_pixel_value: float,
) -> int:
    for p in (ct, us_reference):
        if not p.is_file():
            logger.error("Not a file: %s", p)
            return 1
    if tfm is not None and not tfm.is_file():
        logger.error("Not a file: %s", tfm)
        return 1
    if report_json is not None and not report_json.is_file():
        logger.error("Not a file: %s", report_json)
        return 1

    out.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading CT: %s", ct)
    moving = sitk.ReadImage(str(ct))
    logger.info("Loading US reference: %s", us_reference)
    fixed = sitk.ReadImage(str(us_reference))

    if tfm is not None:
        logger.info("Loading transform: %s", tfm)
        tx = sitk.ReadTransform(str(tfm))
    else:
        assert report_json is not None
        logger.info("Loading transform from JSON: %s", report_json)
        tx = _load_transform_from_report_json(report_json)

    resampled = _resample_moving_to_fixed(
        moving,
        fixed,
        tx,
        interpolator=sitk.sitkLinear,
        default_value=default_pixel_value,
    )
    resampled = _maybe_float32_ct(sitk, resampled)
    # Gzip of multi-GB volumes often hits NWAD / quota / NFS errors; writer flag False is reliable.
    _write_nifti_checked(sitk, resampled, out, use_compression=False)
    logger.info("Wrote %s", out)
    return 0


def _is_integer_pixel_id(sitk, pixel_id) -> bool:
    return pixel_id in (
        sitk.sitkUInt8,
        sitk.sitkInt8,
        sitk.sitkUInt16,
        sitk.sitkInt16,
        sitk.sitkUInt32,
        sitk.sitkInt32,
        sitk.sitkLabelUInt8,
        sitk.sitkLabelUInt16,
        sitk.sitkLabelUInt32,
    )


def _cast_mask_like_moving(sitk, resampled, moving):
    """
    Cast resampled mask to the moving image's integer type.
    Integer resampled output (e.g. nearest-neighbor on UInt8) is only cast;
    float resampled output is rounded first (SimpleITK Round does not support UInt8).
    """
    out_pid = moving.GetPixelID()
    in_pid = resampled.GetPixelID()
    if _is_integer_pixel_id(sitk, in_pid):
        if in_pid == out_pid:
            return resampled
        return sitk.Cast(resampled, out_pid)
    as_float = sitk.Cast(resampled, sitk.sitkFloat32)
    rounded = sitk.Round(as_float)
    if _is_integer_pixel_id(sitk, out_pid):
        return sitk.Cast(rounded, out_pid)
    return sitk.Cast(rounded, sitk.sitkUInt8)


def _run_batch(
    sitk,
    batch_root: Path,
    out_dir: Path,
    ct_images_subdir: str,
    ct_masks_subdir: str,
    us_masks_subdir: str,
    init_transf_subdir: str,
    tfm_basename_template: str,
    default_pixel_value: float,
    strict: bool,
    nifti_suffix: str,
    nifti_gzip: bool,
) -> int:
    ct_img_dir = batch_root / ct_images_subdir
    ct_mask_dir = batch_root / ct_masks_subdir
    us_mask_dir = batch_root / us_masks_subdir
    tfm_dir = batch_root / init_transf_subdir

    ct_by_case = _collect_single_file_per_case(ct_img_dir, "CT_images")
    mask_by_case = _collect_single_file_per_case(ct_mask_dir, "CT_masks")
    us_by_case = _collect_single_file_per_case(us_mask_dir, "US_masks")

    keys_ct, keys_mask, keys_us = ct_by_case.keys(), mask_by_case.keys(), us_by_case.keys()
    union = keys_ct | keys_mask | keys_us
    complete_keys = keys_ct & keys_mask & keys_us
    for case_id in sorted(union - complete_keys):
        missing = []
        if case_id not in keys_ct:
            missing.append(ct_images_subdir)
        if case_id not in keys_mask:
            missing.append(ct_masks_subdir)
        if case_id not in keys_us:
            missing.append(us_masks_subdir)
        logger.warning("Incomplete case %s under %s (missing %s)", case_id, batch_root, ", ".join(missing))

    case_ids = sorted(complete_keys)
    if not case_ids:
        logger.error("No cases with paired CT_images, CT_masks, and US_masks under %s", batch_root)
        return 1

    out_img_dir = out_dir / "resampled_CT_images"
    out_msk_dir = out_dir / "resampled_CT_masks"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_msk_dir.mkdir(parents=True, exist_ok=True)

    use_compression = bool(nifti_gzip and nifti_suffix == ".nii.gz")
    if nifti_suffix == ".nii.gz" and not nifti_gzip:
        logger.warning(
            "batch outputs use suffix .nii.gz but gzip is off (--batch_nifti_gzip not set); "
            "writing without gzip compression. Prefer --batch_nifti_suffix .nii for clarity."
        )

    successes = 0
    failures = 0

    for case_id in case_ids:
        tfm_name = tfm_basename_template.format(case_id=case_id)
        tfm_path = tfm_dir / tfm_name
        if not tfm_path.is_file():
            msg = f"Missing transform for case {case_id}: {tfm_path}"
            if strict:
                logger.error(msg)
                failures += 1
            else:
                logger.warning(msg)
            continue

        ct_path = ct_by_case[case_id]
        mask_path = mask_by_case[case_id]
        us_path = us_by_case[case_id]

        try:
            logger.info("Case %s: loading US reference %s", case_id, us_path.name)
            fixed = sitk.ReadImage(str(us_path))
            tx = sitk.ReadTransform(str(tfm_path))

            stem = _nifti_stem(ct_path)
            out_ct = out_img_dir / f"{stem}_in_US_space{nifti_suffix}"
            logger.info("Case %s: resampling CT image %s", case_id, ct_path.name)
            moving_ct = sitk.ReadImage(str(ct_path))
            res_ct = _resample_moving_to_fixed(
                moving_ct,
                fixed,
                tx,
                interpolator=sitk.sitkLinear,
                default_value=default_pixel_value,
            )
            res_ct = _maybe_float32_ct(sitk, res_ct)
            _write_nifti_checked(sitk, res_ct, out_ct, use_compression=use_compression)

            out_mask = out_msk_dir / f"{_nifti_stem(mask_path)}_in_US_space{nifti_suffix}"
            logger.info("Case %s: resampling CT mask %s", case_id, mask_path.name)
            moving_m = sitk.ReadImage(str(mask_path))
            res_m = _resample_moving_to_fixed(
                moving_m,
                fixed,
                tx,
                interpolator=sitk.sitkNearestNeighbor,
                default_value=0.0,
            )
            res_m = _cast_mask_like_moving(sitk, res_m, moving_m)
            _write_nifti_checked(sitk, res_m, out_mask, use_compression=use_compression)

            logger.info("Case %s: wrote %s and %s", case_id, out_ct.name, out_mask.name)
            successes += 1
        except Exception as exc:
            failures += 1
            logger.exception("Case %s failed: %s", case_id, exc)
            if strict:
                return 1

    if successes == 0:
        logger.error("No cases completed successfully (failures=%d)", failures)
        return 1
    if failures and strict:
        return 1
    logger.info("Batch finished: %d succeeded, %d failed/skipped", successes, failures)
    return 0


def main() -> int:
    import SimpleITK as sitk

    parser = argparse.ArgumentParser(
        description="Resample CT onto US grid using a saved CT->US transform (.tfm or JSON)."
    )
    parser.add_argument(
        "--batch_root",
        type=Path,
        default=None,
        help=(
            "TRUSTED-style dataset root containing CT_images, CT_masks, US_masks, init_transf. "
            "When set, runs batch resampling (requires --out_dir); do not pass single-case paths."
        ),
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Batch mode: output root for resampled_CT_images/ and resampled_CT_masks/.",
    )
    parser.add_argument(
        "--ct_images_subdir",
        type=str,
        default="CT_images",
        help="Batch mode: subdirectory under --batch_root for CT volumes (default: CT_images).",
    )
    parser.add_argument(
        "--ct_masks_subdir",
        type=str,
        default="CT_masks",
        help="Batch mode: subdirectory under --batch_root for CT masks (default: CT_masks).",
    )
    parser.add_argument(
        "--us_masks_subdir",
        type=str,
        default="US_masks",
        help="Batch mode: subdirectory under --batch_root for US masks (fixed grid).",
    )
    parser.add_argument(
        "--init_transf_subdir",
        type=str,
        default="init_transf",
        help="Batch mode: subdirectory under --batch_root for .tfm files (default: init_transf).",
    )
    parser.add_argument(
        "--tfm_basename_template",
        type=str,
        default="{case_id}_CT_to_US_pca_icp.tfm",
        help="Batch mode: transform filename under init_transf; must include {case_id} (default: PCA+ICP name).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Batch mode: exit with error on first failure or if any case is skipped (e.g. missing .tfm).",
    )
    parser.add_argument(
        "--batch_nifti_suffix",
        type=str,
        default=".nii",
        choices=(".nii", ".nii.gz"),
        help=(
            "Batch mode: output filename suffix. Default .nii (uncompressed; avoids ITK NWAD / quota "
            "failures on large gzip writes)."
        ),
    )
    parser.add_argument(
        "--batch_nifti_gzip",
        action="store_true",
        help=(
            "Batch mode: gzip outputs when using --batch_nifti_suffix .nii.gz (large volumes may fail "
            "if disk quota or filesystem is tight)."
        ),
    )

    parser.add_argument("--ct", type=Path, default=None, help="Moving CT volume (.nii / .nii.gz).")
    parser.add_argument(
        "--us_reference",
        type=Path,
        default=None,
        help="Fixed image defining output grid (US mask or US image).",
    )
    parser.add_argument("--out", type=Path, default=None, help="Output resampled CT path (single-case mode).")
    src = parser.add_mutually_exclusive_group(required=False)
    src.add_argument(
        "--tfm",
        type=Path,
        default=None,
        help="CT->US transform from --save_tfm (e.g. *CT_to_US_pca_icp.tfm).",
    )
    src.add_argument(
        "--report_json",
        type=Path,
        default=None,
        help="PCA+ICP report JSON; uses resample_matrix (RAS CT->US).",
    )
    parser.add_argument(
        "--default_pixel_value",
        type=float,
        default=-1000.0,
        help="Value for voxels outside the moving CT (typical HU for air). Default: -1000.",
    )
    args = parser.parse_args()

    batch = args.batch_root is not None
    if batch:
        if args.out_dir is None:
            logger.error("Batch mode requires --out_dir")
            return 1
        if any(
            x is not None
            for x in (args.ct, args.us_reference, args.out, args.tfm, args.report_json)
        ):
            logger.error(
                "Batch mode: do not pass --ct, --us_reference, --out, --tfm, or --report_json "
                "(use --batch_root and --out_dir only for transform source)."
            )
            return 1
        if "{case_id}" not in args.tfm_basename_template:
            logger.error("--tfm_basename_template must contain '{case_id}' placeholder")
            return 1
        if not args.batch_root.is_dir():
            logger.error("Not a directory: %s", args.batch_root)
            return 1
        return _run_batch(
            sitk,
            args.batch_root,
            args.out_dir,
            args.ct_images_subdir,
            args.ct_masks_subdir,
            args.us_masks_subdir,
            args.init_transf_subdir,
            args.tfm_basename_template,
            args.default_pixel_value,
            args.strict,
            args.batch_nifti_suffix,
            args.batch_nifti_gzip,
        )

    if args.ct is None or args.us_reference is None or args.out is None:
        logger.error("Single-case mode requires --ct, --us_reference, and --out (or use --batch_root).")
        return 1
    if args.tfm is None and args.report_json is None:
        logger.error("Single-case mode requires --tfm or --report_json")
        return 1
    if args.tfm is not None and args.report_json is not None:
        logger.error("Pass only one of --tfm or --report_json")
        return 1

    return _run_single_case(
        sitk,
        args.ct,
        args.us_reference,
        args.out,
        args.tfm,
        args.report_json,
        args.default_pixel_value,
    )


if __name__ == "__main__":
    sys.exit(main())
