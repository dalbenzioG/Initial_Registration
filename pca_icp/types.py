from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

if TYPE_CHECKING:
    import vtk


@dataclass
class PCAFrame:
    centroid: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray  # Columns are principal axes.


@dataclass
class TransformCandidate:
    name: str
    sign_matrix: np.ndarray
    matrix4x4: np.ndarray
    moved_source: Optional["vtk.vtkPolyData"] = None


@dataclass
class CandidateScore:
    candidate_name: str
    symmetric_mean_distance: float
    hausdorff95: float


@dataclass
class ICPResult:
    matrix4x4: np.ndarray
    iterations: int
    mean_distance: float
    mode: str


@dataclass
class RegistrationConfig:
    source_label: int = 1
    target_label: int = 1
    smoothing_iterations: int = 10
    decimation_reduction: float = 0.0
    pca_unstable_threshold: float = 1.10
    icp_mode: str = "rigid"
    icp_max_iterations: int = 100
    icp_max_landmarks: int = 1000
    icp_max_mean_distance: float = 1e-3
    run_multistart_on_unstable_pca: bool = True
    multistart_top_k: int = 2


@dataclass
class RegistrationResult:
    source_polydata: "vtk.vtkPolyData"
    target_polydata: "vtk.vtkPolyData"
    registered_source_polydata: "vtk.vtkPolyData"
    pca_matrix: np.ndarray
    icp_matrix: np.ndarray
    final_matrix: np.ndarray
    best_candidate_name: str
    best_candidate_score: float
    candidate_scores: list[CandidateScore]
    diagnostics: dict[str, Any] = field(default_factory=dict)
