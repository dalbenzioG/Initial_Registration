from __future__ import annotations

import numpy as np
import vtk

from .io_vtk import vtk_points_to_numpy
from .types import PCAFrame, TransformCandidate


def compute_pca_frame(polydata: vtk.vtkPolyData) -> PCAFrame:
    points = vtk_points_to_numpy(polydata.GetPoints())
    if points.shape[0] < 3:
        raise ValueError("Need at least 3 points for PCA frame.")

    centroid = points.mean(axis=0)
    centered = points - centroid
    cov = centered.T @ centered / max(len(points) - 1, 1)

    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]

    if np.linalg.det(evecs) < 0:
        evecs[:, 2] *= -1.0

    return PCAFrame(centroid=centroid, eigenvalues=evals, eigenvectors=evecs)


def pca_unstable(evals: np.ndarray, threshold: float = 1.10) -> bool:
    r01 = evals[0] / max(evals[1], 1e-12)
    r12 = evals[1] / max(evals[2], 1e-12)
    return (r01 < threshold) or (r12 < threshold)


def _build_rigid_matrix(rotation: np.ndarray, c_src: np.ndarray, c_tgt: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = c_tgt - rotation @ c_src
    return matrix


def generate_pca_candidates(
    source_poly: vtk.vtkPolyData,
    target_poly: vtk.vtkPolyData,
) -> tuple[list[TransformCandidate], dict[str, object]]:
    source_frame = compute_pca_frame(source_poly)
    target_frame = compute_pca_frame(target_poly)

    flips = [
        np.diag([1.0, 1.0, 1.0]),
        np.diag([1.0, -1.0, -1.0]),
        np.diag([-1.0, 1.0, -1.0]),
        np.diag([-1.0, -1.0, 1.0]),
    ]

    candidates: list[TransformCandidate] = []
    for idx, flip in enumerate(flips):
        rotation = target_frame.eigenvectors @ flip @ source_frame.eigenvectors.T
        det = np.linalg.det(rotation)
        if det < 0:
            continue

        candidates.append(
            TransformCandidate(
                name=f"candidate_{idx}",
                sign_matrix=flip,
                matrix4x4=_build_rigid_matrix(rotation, source_frame.centroid, target_frame.centroid),
            )
        )

    diagnostics = {
        "source_eigenvalues": source_frame.eigenvalues,
        "target_eigenvalues": target_frame.eigenvalues,
        "source_pca_centroid": source_frame.centroid,
        "target_pca_centroid": target_frame.centroid,
        "source_right_handed_det": float(np.linalg.det(source_frame.eigenvectors)),
        "target_right_handed_det": float(np.linalg.det(target_frame.eigenvectors)),
    }
    return candidates, diagnostics
