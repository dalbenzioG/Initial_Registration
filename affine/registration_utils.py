"""
Reusable utilities for CT-US kidney registration using SimpleITK.

Provides pairing, metadata validation, SDM generation, rigid/affine registration,
and resampling. Used by validate_and_generate_sdm.py and run_affine_registration.py.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import SimpleITK as sitk

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Naming patterns (TRUSTED dataset)
# Masks: {patient}{side}_seg.nii.gz, {patient}{side}_maskUS.nii.gz, {patient}{side}_mask.nii.gz
# Images: {patient}{side}_imgCT.nii.gz, {patient}{side}_imgUS.nii.gz
# -----------------------------------------------------------------------------

MASK_SUFFIXES = [
    ("L_seg", "L"),
    ("R_seg", "R"),
    ("L_maskUS", "L"),
    ("R_maskUS", "R"),
    ("L_mask", "L"),
    ("R_mask", "R"),
]
IMAGE_SUFFIXES = [
    ("L_imgCT", "L", "ct"),
    ("R_imgCT", "R", "ct"),
    ("L_imgUS", "L", "us"),
    ("R_imgUS", "R", "us"),
]
CT_IMAGE_SUFFIXES = [(s[0], s[1]) for s in IMAGE_SUFFIXES if s[2] == "ct"]
US_IMAGE_SUFFIXES = [(s[0], s[1]) for s in IMAGE_SUFFIXES if s[2] == "us"]


def extract_patient_side(filename: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract (patient_id, side) from TRUSTED mask or image filename.

    Handles: *_seg, *_maskUS, *_mask (masks); *_imgCT, *_imgUS (images).
    Returns (patient_id, side) e.g. ("200", "L") for "200L_seg.nii.gz".
    """
    base = Path(filename).stem
    if base.endswith(".nii"):
        base = base[:-4]
    for suffix, side in MASK_SUFFIXES + [(s[0], s[1]) for s in IMAGE_SUFFIXES]:
        if base.endswith(suffix):
            patient_id = base[: -len(suffix)].strip("_")
            return patient_id, side
    return None, None


def kidney_key(patient_id: str, side: str) -> str:
    """Produce kidney key e.g. '200L', '200R'."""
    return f"{patient_id}{side}"


def parse_kidney_key(key: str) -> Tuple[str, str]:
    """Parse kidney key '200L' -> ('200', 'L')."""
    if len(key) >= 2 and key[-1] in "LR":
        return key[:-1], key[-1]
    return key, ""


def find_files_by_suffixes(
    directory: Path, suffixes: Sequence[Tuple[str, ...]], ext: str = ".nii.gz"
) -> dict[str, Path]:
    """
    Scan directory for files matching {patient}{side}{suffix}{ext}.
    Returns dict mapping kidney_key -> path.
    """
    result: dict[str, Path] = {}
    if not directory.exists():
        return result
    for f in directory.iterdir():
        if not f.is_file() or ext not in f.name:
            continue
        base = f.stem
        if base.endswith(".nii"):
            base = base[:-4]
        for parts in suffixes:
            suffix = parts[0]
            side = parts[1]
            if base.endswith(suffix):
                patient_id = base[: -len(suffix)].strip("_")
                if patient_id and side:
                    k = kidney_key(patient_id, side)
                    # Prefer first match; masks may have multiple patterns
                    if k not in result:
                        result[k] = f.resolve()
                break
    return result


def collect_mask_image_pairs(
    ct_mask_dir: Path,
    us_mask_dir: Path,
    ct_img_dir: Path,
    us_img_dir: Path,
) -> list[dict]:
    """
    Pair masks and images by kidney key. Returns list of dicts with keys:
    kidney_key, ct_mask_path, us_mask_path, ct_img_path, us_img_path.
    Entries may have None for missing paths.
    """
    ct_masks = find_files_by_suffixes(ct_mask_dir, MASK_SUFFIXES)
    us_masks = find_files_by_suffixes(us_mask_dir, MASK_SUFFIXES)
    ct_imgs = find_files_by_suffixes(ct_img_dir, CT_IMAGE_SUFFIXES)
    us_imgs = find_files_by_suffixes(us_img_dir, US_IMAGE_SUFFIXES)

    all_keys = sorted(set(ct_masks) | set(us_masks) | set(ct_imgs) | set(us_imgs))
    pairs = []
    for key in all_keys:
        pairs.append(
            {
                "kidney_key": key,
                "ct_mask_path": ct_masks.get(key),
                "us_mask_path": us_masks.get(key),
                "ct_img_path": ct_imgs.get(key),
                "us_img_path": us_imgs.get(key),
            }
        )
    return pairs


def collect_sdm_image_pairs(
    sdm_dir: Path,
    ct_img_dir: Path,
    us_img_dir: Path,
    sdm_suffix_ct: str = "imgCT_sdm",
    sdm_suffix_us: str = "imgUS_sdm",
) -> list[dict]:
    """
    Pair SDMs and images by kidney key. Expects SDM names: {key}_{suffix}.nii.gz
    (e.g. 200L_imgCT_sdm.nii.gz, 200L_imgUS_sdm.nii.gz).
    Returns list of dicts with ct_sdm_path, us_sdm_path, ct_img_path, us_img_path.
    """
    ct_imgs = find_files_by_suffixes(ct_img_dir, CT_IMAGE_SUFFIXES)
    us_imgs = find_files_by_suffixes(us_img_dir, US_IMAGE_SUFFIXES)
    pairs = []
    for key in sorted(set(ct_imgs) & set(us_imgs)):
        ct_sdm = sdm_dir / f"{key}_{sdm_suffix_ct}.nii.gz"
        if not ct_sdm.exists():
            ct_sdm = sdm_dir / f"{key}_{sdm_suffix_ct}.nii"
        us_sdm = sdm_dir / f"{key}_{sdm_suffix_us}.nii.gz"
        if not us_sdm.exists():
            us_sdm = sdm_dir / f"{key}_{sdm_suffix_us}.nii"
        if ct_sdm.exists() and us_sdm.exists():
            pairs.append(
                {
                    "kidney_key": key,
                    "ct_sdm_path": ct_sdm,
                    "us_sdm_path": us_sdm,
                    "ct_img_path": ct_imgs[key],
                    "us_img_path": us_imgs[key],
                }
            )
    return pairs


# -----------------------------------------------------------------------------
# Image / mask loading
# -----------------------------------------------------------------------------


def load_binary_mask(path: Path | str) -> sitk.Image:
    """Load mask and binarize (>0 -> 1)."""
    mask = sitk.ReadImage(str(path), sitk.sitkUInt8)
    mask = sitk.Cast(mask > 0, sitk.sitkUInt8)
    return mask


def load_image(path: Path | str) -> sitk.Image:
    """Load image as float32 for registration."""
    img = sitk.ReadImage(str(path))
    return sitk.Cast(img, sitk.sitkFloat32)


def mask_is_empty(mask: sitk.Image) -> bool:
    """Check if mask has no foreground voxels."""
    arr = sitk.GetArrayViewFromImage(mask)
    return bool(arr.sum() == 0)


# -----------------------------------------------------------------------------
# Metadata validation
# -----------------------------------------------------------------------------


@dataclass
class MetadataCheck:
    """Result of metadata comparison."""

    origin_ok: bool
    spacing_ok: bool
    direction_ok: bool
    size_ok: bool
    origin_diff: Optional[Tuple[float, ...]] = None
    spacing_diff: Optional[Tuple[float, ...]] = None

    @property
    def ok(self) -> bool:
        return self.origin_ok and self.spacing_ok and self.direction_ok and self.size_ok


def check_metadata_equality(
    img1: sitk.Image,
    img2: sitk.Image,
    check_size: bool = True,
    origin_tol: float = 1e-6,
    spacing_tol: float = 1e-6,
) -> MetadataCheck:
    """
    Compare origin, spacing, direction, and optionally size of two images.
    """
    o1, o2 = img1.GetOrigin(), img2.GetOrigin()
    s1, s2 = img1.GetSpacing(), img2.GetSpacing()
    d1, d2 = img1.GetDirection(), img2.GetDirection()
    origin_ok = np.allclose(o1, o2, atol=origin_tol, rtol=0)
    spacing_ok = np.allclose(s1, s2, atol=spacing_tol, rtol=0)
    direction_ok = np.allclose(np.array(d1), np.array(d2), atol=1e-6)
    origin_diff = tuple(float(a - b) for a, b in zip(o1, o2)) if not origin_ok else None
    spacing_diff = tuple(float(a - b) for a, b in zip(s1, s2)) if not spacing_ok else None
    size_ok = True
    if check_size:
        size_ok = img1.GetSize() == img2.GetSize()
    return MetadataCheck(
        origin_ok=origin_ok,
        spacing_ok=spacing_ok,
        direction_ok=direction_ok,
        size_ok=size_ok,
        origin_diff=origin_diff,
        spacing_diff=spacing_diff,
    )


def copy_metadata_from_reference(mask: sitk.Image, reference: sitk.Image) -> sitk.Image:
    """
    Copy origin, spacing, direction from reference to mask. Mask size unchanged;
    use only when mask and reference are known to share the same grid.
    """
    out = sitk.Image(mask)
    out.SetOrigin(reference.GetOrigin())
    out.SetSpacing(reference.GetSpacing())
    out.SetDirection(reference.GetDirection())
    return out


# -----------------------------------------------------------------------------
# Centroids and SDMs
# -----------------------------------------------------------------------------


def mask_centroid_mm(mask: sitk.Image) -> Tuple[float, float, float]:
    """Compute centroid of binary mask in physical mm (LabelShapeStatistics, label 1)."""
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(mask)
    return tuple(stats.GetCentroid(1))


def centroid_distance_mm(
    a: Tuple[float, float, float], b: Tuple[float, float, float]
) -> float:
    """Euclidean distance between two points in mm."""
    return float(np.linalg.norm(np.array(a) - np.array(b)))


def mask_to_signed_distance_map(mask: sitk.Image) -> sitk.Image:
    """
    Convert binary mask to signed distance map (Euclidean, in mm).
    Inside=negative, outside=positive (insideIsPositive=False).
    """
    sdf = sitk.SignedMaurerDistanceMap(
        mask,
        insideIsPositive=False,
        squaredDistance=False,
        useImageSpacing=True,
        backgroundValue=0.0,
    )
    return sitk.Cast(sdf, sitk.sitkFloat32)


# -----------------------------------------------------------------------------
# Transform utilities
# -----------------------------------------------------------------------------


def _unwrap_composite(tx: sitk.Transform) -> sitk.Transform:
    """Extract the actual transform from SimpleITK 2.x CompositeTransform."""
    if hasattr(tx, "GetNumberOfTransforms"):
        n = tx.GetNumberOfTransforms()
        if n > 0:
            return tx.GetNthTransform(n - 1)
    return tx


def rigid_to_affine(rigid_tx: sitk.Transform) -> sitk.AffineTransform:
    """Convert rigid/similarity transform to AffineTransform (3D)."""
    rigid_tx = _unwrap_composite(rigid_tx)
    affine = sitk.AffineTransform(3)
    affine.SetMatrix(rigid_tx.GetMatrix())
    affine.SetTranslation(rigid_tx.GetTranslation())
    if hasattr(rigid_tx, "GetCenter"):
        affine.SetCenter(rigid_tx.GetCenter())
    return affine


# -----------------------------------------------------------------------------
# Registration
# -----------------------------------------------------------------------------


def _log_registration_progress(method: sitk.ImageRegistrationMethod) -> None:
    """Callback to log registration iterations."""
    it = method.GetOptimizerIteration()
    if it == 0:
        try:
            scales = method.GetOptimizerScales()
            logger.debug("Optimizer scales: %s", scales)
        except Exception:
            pass
    if it % 50 == 0:
        logger.debug(
            "Iter %d metric=%.6f",
            it,
            method.GetMetricValue(),
        )


def compute_moments_initializer(
    fixed: sitk.Image, moving: sitk.Image, transform: sitk.Transform
) -> sitk.Transform:
    """
    Initialize transform using MOMENTS (center of mass alignment).
    Modifies transform in place; returns it.
    """
    return sitk.CenteredTransformInitializer(
        fixed,
        moving,
        transform,
        sitk.CenteredTransformInitializerFilter.MOMENTS,
    )


def run_rigid_registration(
    fixed: sitk.Image,
    moving: sitk.Image,
    initial_transform: Optional[sitk.Transform] = None,
    num_iterations: int = 200,
    learning_rate: float = 1.0,
    shrink_factors: Sequence[int] = (4, 2, 1),
    smoothing_sigmas: Sequence[float] = (2.0, 1.0, 0.0),
    use_physical_shift: bool = True,
) -> sitk.Transform:
    """
    Rigid registration on SDMs with MeanSquares metric.

    SimpleITK.Execute returns transform mapping fixed -> moving (CT -> US).
    To get US -> CT, use GetInverse().

    Returns: transform T such that T(fixed_physical) = moving_physical.
    """
    tx = sitk.Euler3DTransform()
    if initial_transform is not None:
        tx = sitk.Euler3DTransform(initial_transform)
    else:
        tx = compute_moments_initializer(fixed, moving, tx)

    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMeanSquares()
    R.SetOptimizerAsRegularStepGradientDescent(
        learningRate=learning_rate,
        minStep=1e-4,
        numberOfIterations=num_iterations,
        gradientMagnitudeTolerance=1e-8,
    )
    if use_physical_shift:
        R.SetOptimizerScalesFromPhysicalShift()
    R.SetInterpolator(sitk.sitkLinear)
    R.SetInitialTransform(tx, inPlace=False)
    R.SetShrinkFactorsPerLevel(list(shrink_factors))
    R.SetSmoothingSigmasPerLevel(list(smoothing_sigmas))
    R.SetSmoothingSigmasAreSpecifiedInPhysicalUnits(True)

    out_tx = R.Execute(fixed, moving)
    logger.info(
        "Rigid: %s iterations, metric=%.6f",
        R.GetOptimizerIteration(),
        R.GetMetricValue(),
    )
    return out_tx


def run_affine_registration(
    fixed: sitk.Image,
    moving: sitk.Image,
    initial_transform: Optional[sitk.Transform] = None,
    num_iterations: int = 300,
    learning_rate: float = 1.0,
    shrink_factors: Sequence[int] = (4, 2, 1),
    smoothing_sigmas: Sequence[float] = (2.0, 1.0, 0.0),
    use_physical_shift: bool = True,
) -> sitk.Transform:
    """
    Affine registration on SDMs with MeanSquares metric.

    SimpleITK.Execute returns transform mapping fixed -> moving (CT -> US).
    To resample US into CT space, use tx_us_to_ct = result.GetInverse().

    Returns: transform T such that T(fixed_physical) = moving_physical.
    """
    if initial_transform is not None:
        affine = rigid_to_affine(initial_transform)
    else:
        affine = sitk.AffineTransform(3)
        affine = compute_moments_initializer(fixed, moving, affine)

    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMeanSquares()
    R.SetOptimizerAsRegularStepGradientDescent(
        learningRate=learning_rate,
        minStep=1e-4,
        numberOfIterations=num_iterations,
        gradientMagnitudeTolerance=1e-8,
    )
    if use_physical_shift:
        R.SetOptimizerScalesFromPhysicalShift()
    R.SetInterpolator(sitk.sitkLinear)
    R.SetInitialTransform(affine, inPlace=False)
    R.SetShrinkFactorsPerLevel(list(shrink_factors))
    R.SetSmoothingSigmasPerLevel(list(smoothing_sigmas))
    R.SetSmoothingSigmasAreSpecifiedInPhysicalUnits(True)

    out_tx = R.Execute(fixed, moving)
    logger.info(
        "Affine: %s iterations, metric=%.6f",
        R.GetOptimizerIteration(),
        R.GetMetricValue(),
    )
    return out_tx


def run_multi_stage_registration(
    fixed_sdm: sitk.Image,
    moving_sdm: sitk.Image,
    shrink_factors: Sequence[int] = (4, 2, 1),
    smoothing_sigmas: Sequence[float] = (2.0, 1.0, 0.0),
    rigid_iterations: int = 200,
    affine_iterations: int = 300,
) -> Tuple[sitk.Transform, sitk.Transform]:
    """
    MOMENTS -> rigid -> affine pipeline on SDMs.

    Returns (tx_us_to_ct, tx_ct_to_us) where:
    - tx_us_to_ct: maps US physical coords -> CT physical coords (use to resample US into CT)
    - tx_ct_to_us: maps CT physical coords -> US physical coords
    """
    # Stage 1: MOMENTS + rigid
    tx_ct_to_us = run_rigid_registration(
        fixed_sdm,
        moving_sdm,
        initial_transform=None,
        num_iterations=rigid_iterations,
        shrink_factors=shrink_factors,
        smoothing_sigmas=smoothing_sigmas,
    )
    # Stage 2: affine initialized from rigid
    tx_ct_to_us = run_affine_registration(
        fixed_sdm,
        moving_sdm,
        initial_transform=tx_ct_to_us,
        num_iterations=affine_iterations,
        shrink_factors=shrink_factors,
        smoothing_sigmas=smoothing_sigmas,
    )
    tx_us_to_ct = tx_ct_to_us.GetInverse()
    return tx_us_to_ct, tx_ct_to_us


# -----------------------------------------------------------------------------
# Resampling and I/O
# -----------------------------------------------------------------------------


def resample_to_fixed(
    moving: sitk.Image,
    fixed: sitk.Image,
    tx_moving_to_fixed: sitk.Transform,
    interpolator: int = sitk.sitkLinear,
    default_value: float = 0.0,
) -> sitk.Image:
    """
    Resample moving image into fixed image space.

    tx_moving_to_fixed: transform mapping moving_physical -> fixed_physical
    (e.g. tx_us_to_ct when resampling US into CT).

    ResampleImageFilter expects transform mapping output (fixed) -> input (moving),
    so we pass tx_moving_to_fixed.GetInverse() internally.
    """
    tx_fixed_to_moving = tx_moving_to_fixed.GetInverse()
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetTransform(tx_fixed_to_moving)
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(default_value)
    return resampler.Execute(moving)


def save_transform(transform: sitk.Transform, path: Path | str) -> None:
    """Save transform to file."""
    sitk.WriteTransform(transform, str(path))
