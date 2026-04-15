from __future__ import annotations

import os
import re


def extract_case_id(filename: str) -> str | None:
    """
    Extract case ID like "314L" from names such as:
    - 314L_maskCT.nii
    - 314L_maskUS.nii
    """
    match = re.match(r"(\d+[LR])_", filename)
    if match:
        return match.group(1)
    return None


def build_dataset_index(base_dir: str) -> dict[str, dict[str, str]]:
    """
    Build mapping of paired masks from:
      {base_dir}/CT_masks and {base_dir}/US_masks.
    """
    ct_dir = os.path.join(base_dir, "CT_masks")
    us_dir = os.path.join(base_dir, "US_masks")

    dataset: dict[str, dict[str, str]] = {}

    for filename in os.listdir(ct_dir):
        case_id = extract_case_id(filename)
        if case_id is None:
            continue
        dataset.setdefault(case_id, {})
        dataset[case_id]["ct_mask"] = os.path.join(ct_dir, filename)

    for filename in os.listdir(us_dir):
        case_id = extract_case_id(filename)
        if case_id is None:
            continue
        dataset.setdefault(case_id, {})
        dataset[case_id]["us_mask"] = os.path.join(us_dir, filename)

    valid_cases: dict[str, dict[str, str]] = {}
    for case_id, paths in dataset.items():
        if "ct_mask" in paths and "us_mask" in paths:
            valid_cases[case_id] = paths
        else:
            print(f"[WARNING] Incomplete pair for {case_id}: {paths}")

    return valid_cases
