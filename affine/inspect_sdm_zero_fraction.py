#!/usr/bin/env python3
"""
Inspect zero-fraction statistics for signed distance maps.

Usage:
  python utils/inspect_sdm_zero_fraction.py --sdm_dir /path/to/sdms --limit 5
  python utils/inspect_sdm_zero_fraction.py --sdm_dir /path/to/sdms --suffix imgUS_sdm --limit 10
"""

import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def inspect_sdm(path: Path) -> dict:
    """Compute zero-fraction stats for a single SDM file."""
    img = sitk.ReadImage(str(path))
    arr = np.asarray(sitk.GetArrayFromImage(img))

    zf_global = float(np.mean(arr == 0))
    zf_axial = np.mean(arr == 0, axis=(1, 2))
    zf_coronal = np.mean(arr == 0, axis=(0, 2))
    zf_sagittal = np.mean(arr == 0, axis=(0, 1))

    return {
        "path": str(path),
        "shape": arr.shape,
        "zf_global": zf_global,
        "zf_axial_max": float(zf_axial.max()),
        "zf_axial_max_slice": int(np.argmax(zf_axial)),
        "zf_coronal_max": float(zf_coronal.max()),
        "zf_coronal_max_slice": int(np.argmax(zf_coronal)),
        "zf_sagittal_max": float(zf_sagittal.max()),
        "zf_sagittal_max_slice": int(np.argmax(zf_sagittal)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect zero-fraction statistics for SDM files"
    )
    parser.add_argument(
        "--sdm_dir",
        type=Path,
        required=True,
        help="Directory containing SDM .nii.gz files",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N SDMs (default: all)",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default=None,
        help="Filter by suffix in filename (e.g. imgUS_sdm for *_imgUS_sdm.nii.gz)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*_sdm.nii.gz",
        help="Glob pattern for SDM files (default: *_sdm.nii.gz)",
    )
    args = parser.parse_args()

    sdm_dir = args.sdm_dir
    if not sdm_dir.exists():
        raise FileNotFoundError(f"SDM directory not found: {sdm_dir}")

    # Collect SDM files
    files = sorted(sdm_dir.glob(args.pattern))
    if args.suffix:
        files = [f for f in files if args.suffix in f.name]
    if args.limit is not None:
        files = files[: args.limit]

    if not files:
        print(f"No SDM files found in {sdm_dir} matching pattern {args.pattern}")
        if args.suffix:
            print(f"  with suffix '{args.suffix}'")
        return

    print(f"Inspecting {len(files)} SDM file(s)\n")

    for path in files:
        try:
            stats = inspect_sdm(path)
        except Exception as e:
            print(f"[ERROR] {path.name}: {e}\n")
            continue

        print(f"--- {path.name} ---")
        print(f"  shape: {stats['shape']}")
        print(f"  global fraction zero: {stats['zf_global']:.4f}")
        print(
            f"  max axial zero fraction: {stats['zf_axial_max']:.4f} at slice {stats['zf_axial_max_slice']}"
        )
        print(
            f"  max coronal zero fraction: {stats['zf_coronal_max']:.4f} at slice {stats['zf_coronal_max_slice']}"
        )
        print(
            f"  max sagittal zero fraction: {stats['zf_sagittal_max']:.4f} at slice {stats['zf_sagittal_max_slice']}"
        )
        print()


if __name__ == "__main__":
    main()
