from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class ManifestCase:
    case_id: str
    tag_path: Path
    moving_image_path: Path | None
    fixed_image_path: Path | None
    raw_payload: dict[str, Any]


def _as_optional_path(value: Any) -> Path | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return None
        return Path(stripped)
    return Path(str(value))


def load_manifest_cases(
    manifest_path: str | Path,
    *,
    cases_key: str = "cases",
    case_id_key: str = "case_id",
    tag_key: str = "tag_path",
    moving_img_key: str = "moving_image_path",
    fixed_img_key: str = "fixed_image_path",
) -> list[ManifestCase]:
    """
    Load case definitions from a JSON manifest.

    Expected shape:
    {
      "cases": [
        {
          "case_id": "...",
          "tag_path": "...",
          "moving_image_path": "...",
          "fixed_image_path": "..."
        }
      ]
    }
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Manifest JSON root must be an object.")

    cases_payload = payload.get(cases_key)
    if not isinstance(cases_payload, list):
        raise ValueError(f"Manifest key '{cases_key}' must contain a list of case objects.")

    cases: list[ManifestCase] = []
    for idx, case_payload in enumerate(cases_payload):
        if not isinstance(case_payload, dict):
            raise ValueError(f"Case at index {idx} must be a JSON object.")

        missing = [
            key for key in (case_id_key, tag_key, moving_img_key, fixed_img_key) if key not in case_payload
        ]
        if missing:
            raise ValueError(f"Case at index {idx} is missing required keys: {missing}")

        case_id_value = case_payload.get(case_id_key)
        if case_id_value is None or str(case_id_value).strip() == "":
            raise ValueError(f"Case at index {idx} has empty case id field '{case_id_key}'.")

        tag_path = _as_optional_path(case_payload.get(tag_key))
        if tag_path is None:
            raise ValueError(f"Case '{case_id_value}' has empty tag path field '{tag_key}'.")

        base_dir = manifest_path.parent
        tag_path = tag_path if tag_path.is_absolute() else (base_dir / tag_path)
        moving_img = _as_optional_path(case_payload.get(moving_img_key))
        fixed_img = _as_optional_path(case_payload.get(fixed_img_key))
        if moving_img is not None and not moving_img.is_absolute():
            moving_img = base_dir / moving_img
        if fixed_img is not None and not fixed_img.is_absolute():
            fixed_img = base_dir / fixed_img

        cases.append(
            ManifestCase(
                case_id=str(case_id_value).strip(),
                tag_path=tag_path,
                moving_image_path=moving_img,
                fixed_image_path=fixed_img,
                raw_payload=case_payload,
            )
        )

    if not cases:
        raise ValueError("Manifest contains zero cases.")
    return cases


def read_mni_tag_file(tag_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Read a 2-volume MNI .tag file.

    Expected point format:
        moving_x moving_y moving_z fixed_x fixed_y fixed_z ""
    """
    tag_path = Path(tag_path)
    if not tag_path.exists():
        raise FileNotFoundError(f"Tag file not found: {tag_path}")

    moving_points: list[list[float]] = []
    fixed_points: list[list[float]] = []
    inside_points = False
    number_pattern = re.compile(r"[-+]?\d*\.\d+|[-+]?\d+")

    with tag_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()

            if line.startswith("Points"):
                inside_points = True
                continue
            if not inside_points:
                continue
            if line == "" or line.startswith("%"):
                continue

            nums = number_pattern.findall(line)
            if len(nums) < 6:
                continue
            values = list(map(float, nums[:6]))
            moving_points.append(values[0:3])
            fixed_points.append(values[3:6])

    moving_arr = np.asarray(moving_points, dtype=np.float64)
    fixed_arr = np.asarray(fixed_points, dtype=np.float64)

    if moving_arr.shape[0] == 0:
        raise ValueError(f"No landmark pairs found in file: {tag_path}")
    if moving_arr.shape != fixed_arr.shape:
        raise ValueError(
            f"Moving and fixed landmark arrays have different shapes: "
            f"{moving_arr.shape} vs {fixed_arr.shape}"
        )
    if moving_arr.ndim != 2 or moving_arr.shape[1] != 3:
        raise ValueError(f"Landmarks must have shape (N, 3), got {moving_arr.shape}")

    return moving_arr, fixed_arr
