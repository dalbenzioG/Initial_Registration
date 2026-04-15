#!/usr/bin/env python3
"""
Run multi-stage affine registration (MOMENTS -> rigid -> affine) on CT-US kidney SDMs.

Uses SDMs for registration metric; resamples US into CT space (unless --transforms_only).
Output: aligned US volume per kidney, and/or transforms in both directions.

Transform directions (saved when --save_transform, ITK "from parent" for Slicer):
  - {key}_US_to_CT_affine.tfm: maps CT->US (put US under transform to show US in CT space)
  - {key}_CT_to_US_affine.tfm: maps US->CT (put CT under transform to show CT in US space)

Usage:
  python utils/run_affine_registration.py \\
    --sdm_dir /path/to/sdms \\
    --ct_img_dir /path/to/CT/images \\
    --us_img_dir /path/to/US/images \\
    --out_dir /path/to/output \\
    [--save_transform] [--transforms_only] [--transform_dir /path/to/transforms]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import SimpleITK as sitk
from tqdm import tqdm

# Ensure repo root is on path when run as script
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.registration_utils import (
    collect_sdm_image_pairs,
    load_image,
    resample_to_fixed,
    run_multi_stage_registration,
    save_transform,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run multi-stage affine registration on CT-US kidney SDMs"
    )
    parser.add_argument("--sdm_dir", type=Path, required=True, help="Directory with SDMs")
    parser.add_argument(
        "--ct_img_dir",
        type=Path,
        required=True,
        help="CT image directory (for reference geometry)",
    )
    parser.add_argument(
        "--us_img_dir",
        type=Path,
        required=True,
        help="US image directory (moving images)",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Output directory for aligned US images (required unless --transforms_only)",
    )
    parser.add_argument(
        "--transforms_only",
        action="store_true",
        help="Only compute and save transforms; skip resampling aligned images",
    )
    parser.add_argument(
        "--save_transform",
        action="store_true",
        help="Save transforms per kidney (saves both US_to_CT and CT_to_US)",
    )
    parser.add_argument(
        "--transform_dir",
        type=Path,
        default=None,
        help="Directory for saved transforms (default: out_dir)",
    )
    parser.add_argument(
        "--rigid_iterations",
        type=int,
        default=200,
        help="Rigid stage iterations per level (default: 200)",
    )
    parser.add_argument(
        "--affine_iterations",
        type=int,
        default=300,
        help="Affine stage iterations per level (default: 300)",
    )
    parser.add_argument(
        "--shrink_factors",
        type=str,
        default="4,2,1",
        help="Multi-resolution shrink factors, comma-separated (default: 4,2,1)",
    )
    parser.add_argument(
        "--smoothing_sigmas",
        type=str,
        default="2.0,1.0,0.0",
        help="Smoothing sigmas per level, comma-separated (default: 2.0,1.0,0.0)",
    )
    parser.add_argument(
        "--sdm_suffix_ct",
        type=str,
        default="imgCT_sdm",
        help="SDM filename suffix for CT (default: imgCT_sdm)",
    )
    parser.add_argument(
        "--sdm_suffix_us",
        type=str,
        default="imgUS_sdm",
        help="SDM filename suffix for US (default: imgUS_sdm)",
    )
    args = parser.parse_args()

    if args.transforms_only:
        if not args.save_transform:
            logger.error("--transforms_only requires --save_transform")
            return 1
        transform_dir = args.transform_dir or args.out_dir
        if transform_dir is None:
            logger.error("--transforms_only requires --transform_dir or --out_dir")
            return 1
    else:
        if args.out_dir is None:
            logger.error("--out_dir is required unless --transforms_only")
            return 1
        args.out_dir.mkdir(parents=True, exist_ok=True)
        transform_dir = args.transform_dir or args.out_dir

    if args.save_transform:
        transform_dir.mkdir(parents=True, exist_ok=True)

    shrink_factors = [int(x.strip()) for x in args.shrink_factors.split(",")]
    smoothing_sigmas = [float(x.strip()) for x in args.smoothing_sigmas.split(",")]

    pairs = collect_sdm_image_pairs(
        args.sdm_dir,
        args.ct_img_dir,
        args.us_img_dir,
        sdm_suffix_ct=args.sdm_suffix_ct,
        sdm_suffix_us=args.sdm_suffix_us,
    )
    if not pairs:
        logger.error("No SDM/image pairs found")
        return 1

    logger.info("Found %d pairs", len(pairs))

    failed = 0
    for pair in tqdm(pairs, desc="Registration"):
        key = pair["kidney_key"]
        ct_sdm_path = pair["ct_sdm_path"]
        us_sdm_path = pair["us_sdm_path"]
        ct_img_path = pair["ct_img_path"]
        us_img_path = pair["us_img_path"]

        try:
            fixed_sdm = sitk.ReadImage(str(ct_sdm_path), sitk.sitkFloat32)
            moving_sdm = sitk.ReadImage(str(us_sdm_path), sitk.sitkFloat32)

            tx_us_to_ct, tx_ct_to_us = run_multi_stage_registration(
                fixed_sdm,
                moving_sdm,
                shrink_factors=shrink_factors,
                smoothing_sigmas=smoothing_sigmas,
                rigid_iterations=args.rigid_iterations,
                affine_iterations=args.affine_iterations,
            )

            if not args.transforms_only:
                fixed_img = load_image(ct_img_path)
                moving_img = load_image(us_img_path)
                aligned_us = resample_to_fixed(
                    moving=moving_img,
                    fixed=fixed_img,
                    tx_moving_to_fixed=tx_us_to_ct,
                    interpolator=sitk.sitkLinear,
                    default_value=0.0,
                )
                out_path = args.out_dir / f"{key}_US_aligned_to_CT.nii.gz"
                sitk.WriteImage(aligned_us, str(out_path))

            if args.save_transform:
                # ITK "from parent" convention for Slicer: file stores parent->child.
                # US_to_CT: put US under transform -> parent=CT, child=US -> store tx_ct_to_us.
                # CT_to_US: put CT under transform -> parent=US, child=CT -> store tx_us_to_ct.
                save_transform(tx_ct_to_us, transform_dir / f"{key}_US_to_CT_affine.tfm")
                save_transform(tx_us_to_ct, transform_dir / f"{key}_CT_to_US_affine.tfm")

        except Exception as e:
            logger.exception("Failed %s: %s", key, e)
            failed += 1

    if failed > 0:
        logger.error("%d pairs failed", failed)
        return 1
    logger.info("All %d pairs completed successfully", len(pairs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
