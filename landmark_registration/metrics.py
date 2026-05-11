from __future__ import annotations

from pathlib import Path

import numpy as np

from .transform import apply_transform


def compute_landmark_errors(
    moving_points: np.ndarray,
    fixed_points: np.ndarray,
    *,
    transform_4x4: np.ndarray | None = None,
) -> dict[str, float | list[float]]:
    """
    Compute Euclidean landmark errors before/after applying transform.
    """
    if transform_4x4 is not None:
        moving_eval = apply_transform(moving_points, transform_4x4)
    else:
        moving_eval = moving_points

    errors = np.linalg.norm(moving_eval - fixed_points, axis=1)
    return {
        "mean": float(np.mean(errors)),
        "std": float(np.std(errors)),
        "min": float(np.min(errors)),
        "max": float(np.max(errors)),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "errors": [float(v) for v in errors],
    }


def _resample_moving_on_fixed_grid(
    moving_data: np.ndarray,
    moving_affine: np.ndarray,
    fixed_shape: tuple[int, int, int],
    fixed_affine: np.ndarray,
    moving_to_fixed_world: np.ndarray,
) -> np.ndarray:
    """
    Resample moving image into fixed voxel grid using linear interpolation.

    World mapping:
      world_fixed = moving_to_fixed_world @ world_moving
    """
    from scipy.ndimage import affine_transform

    moving_to_fixed_idx = np.linalg.inv(fixed_affine) @ moving_to_fixed_world @ moving_affine
    fixed_to_moving_idx = np.linalg.inv(moving_to_fixed_idx)

    matrix = fixed_to_moving_idx[:3, :3]
    offset = fixed_to_moving_idx[:3, 3]
    return affine_transform(
        moving_data.astype(np.float64),
        matrix=matrix,
        offset=offset,
        output_shape=fixed_shape,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )


def _safe_pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = x.reshape(-1)
    y = y.reshape(-1)
    if x.size == 0 or y.size == 0:
        return float("nan")
    x_std = float(np.std(x))
    y_std = float(np.std(y))
    if x_std < 1e-12 or y_std < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _fit_linear_combination(fixed: np.ndarray, moved: np.ndarray) -> np.ndarray:
    grad_z, grad_y, grad_x = np.gradient(moved.astype(np.float64))
    grad_mag = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)

    design = np.stack(
        [
            moved.reshape(-1),
            grad_mag.reshape(-1),
            np.ones(moved.size, dtype=np.float64),
        ],
        axis=1,
    )
    target = fixed.reshape(-1)

    coeffs, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    predicted = design @ coeffs
    return predicted.reshape(fixed.shape)


def compute_lc2_metric(
    moving_image_path: str | Path,
    fixed_image_path: str | Path,
    *,
    moving_to_fixed_world: np.ndarray | None = None,
) -> float:
    """
    Compute LC²: linear correlation between fixed image and a linear
    combination of moved image intensity + gradient magnitude.

    If no transform is provided, moving image is evaluated in its native space
    and must already be aligned with fixed image geometry.
    """
    try:
        import nibabel as nib
    except ImportError as exc:
        raise RuntimeError(
            "LC2 computation requires nibabel. Install with: pip install nibabel"
        ) from exc

    moving_img = nib.load(str(moving_image_path))
    fixed_img = nib.load(str(fixed_image_path))

    moving_data = np.asanyarray(moving_img.dataobj, dtype=np.float64)
    fixed_data = np.asanyarray(fixed_img.dataobj, dtype=np.float64)

    if moving_data.ndim != 3 or fixed_data.ndim != 3:
        raise ValueError(
            f"LC2 expects 3D images. Got moving={moving_data.shape}, fixed={fixed_data.shape}"
        )

    if moving_to_fixed_world is None:
        if moving_data.shape != fixed_data.shape:
            raise ValueError(
                "When no transform is provided, moving and fixed images must have equal shape."
            )
        moved = moving_data
    else:
        moved = _resample_moving_on_fixed_grid(
            moving_data=moving_data,
            moving_affine=moving_img.affine,
            fixed_shape=fixed_data.shape,
            fixed_affine=fixed_img.affine,
            moving_to_fixed_world=moving_to_fixed_world,
        )

    linear_combo = _fit_linear_combination(fixed=fixed_data, moved=moved)
    return _safe_pearson_corr(fixed_data, linear_combo)
