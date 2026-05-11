#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _case_sort_key(case_path: Path) -> tuple[int, str]:
    name = case_path.name
    digits = "".join(ch for ch in name if ch.isdigit())
    number = int(digits) if digits else 10**9
    return number, name.lower()


def _normalize_rel(path: Path) -> str:
    return path.as_posix()


def _select_single_flair(mri_dir: Path) -> list[Path]:
    return [
        p
        for p in sorted(mri_dir.glob("*.nii.gz"))
        if p.is_file() and "flair" in p.name.lower()
    ]


def build_manifest(dataset_root: Path) -> tuple[dict[str, list[dict[str, str]]], list[str]]:
    cases_payload: list[dict[str, str]] = []
    errors: list[str] = []

    if not dataset_root.is_dir():
        raise NotADirectoryError(f"Dataset root does not exist or is not a directory: {dataset_root}")

    case_dirs = sorted(
        [p for p in dataset_root.iterdir() if p.is_dir() and p.name.startswith("Case")],
        key=_case_sort_key,
    )
    if not case_dirs:
        raise ValueError(f"No case folders found under: {dataset_root}")

    base_dir = dataset_root.parent
    for case_dir in case_dirs:
        case_id = case_dir.name
        tag_path = case_dir / "Landmarks" / f"{case_id}-MRI-beforeUS.tag"
        fixed_path = case_dir / "US" / f"{case_id}-US-before.nii.gz"
        mri_dir = case_dir / "MRI"

        flair_candidates = _select_single_flair(mri_dir) if mri_dir.is_dir() else []

        if not tag_path.is_file():
            errors.append(f"{case_id}: missing tag file: {tag_path}")
            continue
        if not fixed_path.is_file():
            errors.append(f"{case_id}: missing US-before file: {fixed_path}")
            continue
        if len(flair_candidates) != 1:
            errors.append(
                f"{case_id}: expected exactly one FLAIR MRI (.nii.gz), found {len(flair_candidates)} "
                f"in {mri_dir}: {[p.name for p in flair_candidates]}"
            )
            continue

        moving_path = flair_candidates[0]
        cases_payload.append(
            {
                "case_id": case_id,
                "tag_path": _normalize_rel(tag_path.relative_to(base_dir)),
                "moving_image_path": _normalize_rel(moving_path.relative_to(base_dir)),
                "fixed_image_path": _normalize_rel(fixed_path.relative_to(base_dir)),
            }
        )

    return {"cases": cases_payload}, errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a landmark_registration manifest for RESECT-style data layout "
            "(Case*/Landmarks, Case*/MRI, Case*/US)."
        )
    )
    parser.add_argument(
        "--dataset_root",
        type=Path,
        required=True,
        help="Path to RESECT-Data directory containing Case folders.",
    )
    parser.add_argument(
        "--out_manifest",
        type=Path,
        default=None,
        help=(
            "Output manifest path. Default: <dataset_root_parent>/resect_landmark_manifest.json"
        ),
    )
    parser.add_argument(
        "--allow_partial",
        action="store_true",
        help=(
            "Write manifest even if some cases fail validation. "
            "Without this flag, any case error exits with code 1 and no file write."
        ),
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    out_manifest = (
        args.out_manifest.resolve()
        if args.out_manifest is not None
        else dataset_root.parent / "resect_landmark_manifest.json"
    )

    try:
        manifest, errors = build_manifest(dataset_root)
    except Exception as exc:
        print(f"Manifest generation failed: {exc}")
        return 1

    if errors:
        print("Validation issues found:")
        for err in errors:
            print(f" - {err}")
        if not args.allow_partial:
            print("No manifest written. Re-run with --allow_partial to write valid cases only.")
            return 1

    if not manifest["cases"]:
        print("No valid cases found; manifest not written.")
        return 1

    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote manifest: {out_manifest}")
    print(f"Cases written: {len(manifest['cases'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
