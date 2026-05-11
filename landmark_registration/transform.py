from __future__ import annotations

from pathlib import Path

import numpy as np

VALID_MODELS = ("rigid", "similarity", "affine")


def compute_landmarks_transform(
    moving_points: np.ndarray,
    fixed_points: np.ndarray,
    *,
    model: str = "rigid",
) -> np.ndarray:
    """
    Compute a 4x4 homogeneous transform mapping moving_points -> fixed_points.
    """
    if model not in VALID_MODELS:
        raise ValueError(f"model must be one of {VALID_MODELS}")
    if moving_points.shape != fixed_points.shape:
        raise ValueError("moving_points and fixed_points must have the same shape.")
    if moving_points.ndim != 2 or moving_points.shape[1] != 3:
        raise ValueError("Landmarks must have shape (N, 3).")
    if moving_points.shape[0] < 3:
        raise ValueError("At least 3 landmark pairs are required.")

    try:
        import vtk
    except ImportError:
        return _compute_landmarks_transform_numpy(moving_points, fixed_points, model=model)

    n_points = moving_points.shape[0]
    vtk_moving_points = vtk.vtkPoints()
    vtk_fixed_points = vtk.vtkPoints()
    vtk_moving_points.SetNumberOfPoints(n_points)
    vtk_fixed_points.SetNumberOfPoints(n_points)

    for i in range(n_points):
        vtk_moving_points.SetPoint(
            i,
            float(moving_points[i, 0]),
            float(moving_points[i, 1]),
            float(moving_points[i, 2]),
        )
        vtk_fixed_points.SetPoint(
            i,
            float(fixed_points[i, 0]),
            float(fixed_points[i, 1]),
            float(fixed_points[i, 2]),
        )

    transform = vtk.vtkLandmarkTransform()
    if model == "rigid":
        transform.SetModeToRigidBody()
    elif model == "similarity":
        transform.SetModeToSimilarity()
    else:
        transform.SetModeToAffine()

    transform.SetSourceLandmarks(vtk_moving_points)
    transform.SetTargetLandmarks(vtk_fixed_points)
    transform.Update()

    vtk_matrix = transform.GetMatrix()
    out = np.eye(4, dtype=np.float64)
    for i in range(4):
        for j in range(4):
            out[i, j] = vtk_matrix.GetElement(i, j)
    return out


def _compute_landmarks_transform_numpy(
    moving_points: np.ndarray,
    fixed_points: np.ndarray,
    *,
    model: str,
) -> np.ndarray:
    if model == "affine":
        return _solve_affine(moving_points, fixed_points)
    if model == "rigid":
        return _solve_umeyama(moving_points, fixed_points, allow_scaling=False)
    return _solve_umeyama(moving_points, fixed_points, allow_scaling=True)


def _solve_affine(moving_points: np.ndarray, fixed_points: np.ndarray) -> np.ndarray:
    x = np.concatenate(
        [moving_points.astype(np.float64), np.ones((moving_points.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    y = fixed_points.astype(np.float64)
    b, _, _, _ = np.linalg.lstsq(x, y, rcond=None)  # (4, 3)

    out = np.eye(4, dtype=np.float64)
    out[:3, :4] = b.T
    return out


def _solve_umeyama(
    moving_points: np.ndarray,
    fixed_points: np.ndarray,
    *,
    allow_scaling: bool,
) -> np.ndarray:
    x = moving_points.astype(np.float64)
    y = fixed_points.astype(np.float64)
    n = x.shape[0]
    if n < 3:
        raise ValueError("At least 3 points are required for rigid/similarity alignment.")

    mu_x = np.mean(x, axis=0)
    mu_y = np.mean(y, axis=0)
    x0 = x - mu_x
    y0 = y - mu_y

    cov = (x0.T @ y0) / float(n)
    u, s, vt = np.linalg.svd(cov)
    d = np.eye(3, dtype=np.float64)
    v = vt.T
    if np.linalg.det(v @ u.T) < 0.0:
        d[-1, -1] = -1.0
    r = v @ d @ u.T

    if allow_scaling:
        var_x = float(np.mean(np.sum(x0**2, axis=1)))
        if var_x < 1e-12:
            raise ValueError("Degenerate point configuration for similarity transform.")
        scale = float(np.trace(np.diag(s) @ d) / var_x)
    else:
        scale = 1.0

    t = mu_y - (scale * r @ mu_x)

    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = scale * r
    out[:3, 3] = t
    return out


def apply_transform(points: np.ndarray, transform_4x4: np.ndarray) -> np.ndarray:
    """
    Apply a homogeneous transform to an (N, 3) array.
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3).")
    if transform_4x4.shape != (4, 4):
        raise ValueError("transform_4x4 must have shape (4, 4).")

    points_h = np.concatenate(
        [points.astype(np.float64), np.ones((points.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    transformed_h = points_h @ transform_4x4.T
    return transformed_h[:, :3]


def save_matrix_txt(transform_4x4: np.ndarray, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_path, transform_4x4, fmt="%.10f")


def _validate_transform_4x4(transform_4x4: np.ndarray) -> np.ndarray:
    matrix = np.asarray(transform_4x4, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"transform_4x4 must have shape (4, 4), got {matrix.shape}.")
    if not np.allclose(matrix[3], np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64), atol=1e-6):
        raise ValueError("transform_4x4 last row must be approximately [0, 0, 0, 1].")
    return matrix


def matrix4x4_to_sitk_affine(transform_4x4: np.ndarray):
    """
    Convert a homogeneous 4x4 moving->fixed matrix to a SimpleITK AffineTransform(3).
    """
    matrix = _validate_transform_4x4(transform_4x4)
    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise ImportError(
            "SimpleITK is required to convert a 4x4 matrix to .tfm. Install with `pip install SimpleITK`."
        ) from exc

    affine = sitk.AffineTransform(3)
    affine.SetMatrix(tuple(float(v) for v in matrix[:3, :3].reshape(-1)))
    affine.SetTranslation(tuple(float(v) for v in matrix[:3, 3]))
    return affine


def save_matrix_tfm(transform_4x4: np.ndarray, output_path: str | Path) -> None:
    """
    Save a moving->fixed homogeneous 4x4 transform as an ITK/SimpleITK .tfm file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    affine = matrix4x4_to_sitk_affine(transform_4x4)
    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise ImportError(
            "SimpleITK is required to write .tfm transforms. Install with `pip install SimpleITK`."
        ) from exc
    sitk.WriteTransform(affine, str(output_path))
