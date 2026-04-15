"""PCA + ICP registration utilities for CT-US kidney segmentations."""

from .dataset import build_dataset_index, extract_case_id
from .types import RegistrationConfig, RegistrationResult


def register_nii_segmentations(*args, **kwargs):
    """Lazy import to avoid hard dependency at package-import time."""
    from .pipeline import register_nii_segmentations as _register

    return _register(*args, **kwargs)


__all__ = [
    "build_dataset_index",
    "extract_case_id",
    "register_nii_segmentations",
    "RegistrationConfig",
    "RegistrationResult",
]
