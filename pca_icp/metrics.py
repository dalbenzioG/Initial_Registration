from __future__ import annotations

import numpy as np
import vtk

from .types import CandidateScore


def _pointwise_abs_distance_to_surface(points_poly: vtk.vtkPolyData, surface_poly: vtk.vtkPolyData) -> np.ndarray:
    implicit_distance = vtk.vtkImplicitPolyDataDistance()
    implicit_distance.SetInput(surface_poly)

    distances = np.empty(points_poly.GetNumberOfPoints(), dtype=np.float64)
    point = [0.0, 0.0, 0.0]
    for i in range(points_poly.GetNumberOfPoints()):
        points_poly.GetPoint(i, point)
        distances[i] = abs(implicit_distance.EvaluateFunction(point))
    return distances


def mean_distance_points_to_surface(points_poly: vtk.vtkPolyData, surface_poly: vtk.vtkPolyData) -> float:
    distances = _pointwise_abs_distance_to_surface(points_poly, surface_poly)
    return float(np.mean(distances)) if len(distances) > 0 else float("inf")


def symmetric_mean_surface_distance(poly_a: vtk.vtkPolyData, poly_b: vtk.vtkPolyData) -> float:
    return 0.5 * (
        mean_distance_points_to_surface(poly_a, poly_b)
        + mean_distance_points_to_surface(poly_b, poly_a)
    )


def hausdorff95(poly_a: vtk.vtkPolyData, poly_b: vtk.vtkPolyData) -> float:
    d_ab = _pointwise_abs_distance_to_surface(poly_a, poly_b)
    d_ba = _pointwise_abs_distance_to_surface(poly_b, poly_a)
    if len(d_ab) == 0 or len(d_ba) == 0:
        return float("inf")
    return float(max(np.percentile(d_ab, 95), np.percentile(d_ba, 95)))


def score_candidate(candidate_name: str, moved_source: vtk.vtkPolyData, target_poly: vtk.vtkPolyData) -> CandidateScore:
    return CandidateScore(
        candidate_name=candidate_name,
        symmetric_mean_distance=symmetric_mean_surface_distance(moved_source, target_poly),
        hausdorff95=hausdorff95(moved_source, target_poly),
    )
