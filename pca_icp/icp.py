from __future__ import annotations

import vtk

from .io_vtk import numpy_from_vtk_matrix
from .types import ICPResult


def run_icp(
    source_poly: vtk.vtkPolyData,
    target_poly: vtk.vtkPolyData,
    mode: str = "rigid",
    max_iterations: int = 100,
    max_landmarks: int = 1000,
    max_mean_distance: float = 1e-3,
) -> ICPResult:
    icp = vtk.vtkIterativeClosestPointTransform()
    icp.SetSource(source_poly)
    icp.SetTarget(target_poly)

    landmark = icp.GetLandmarkTransform()
    if mode == "rigid":
        landmark.SetModeToRigidBody()
    elif mode == "similarity":
        landmark.SetModeToSimilarity()
    elif mode == "affine":
        landmark.SetModeToAffine()
    else:
        raise ValueError(f"Unsupported ICP mode: {mode}")

    icp.SetMaximumNumberOfIterations(max_iterations)
    icp.SetMaximumNumberOfLandmarks(max_landmarks)
    icp.CheckMeanDistanceOn()
    icp.SetMaximumMeanDistance(max_mean_distance)
    icp.StartByMatchingCentroidsOff()
    icp.Modified()
    icp.Update()

    return ICPResult(
        matrix4x4=numpy_from_vtk_matrix(icp.GetMatrix()),
        iterations=icp.GetNumberOfIterations(),
        mean_distance=icp.GetMeanDistance(),
        mode=mode,
    )
