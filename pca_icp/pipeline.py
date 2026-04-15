from __future__ import annotations

from typing import Any

import vtk

from .icp import run_icp
from .io_vtk import load_binary_segmentation_nii_as_polydata
from .metrics import score_candidate
from .pca_init import generate_pca_candidates, pca_unstable
from .preprocess import clean_and_decimate
from .transforms import apply_matrix_to_polydata, compose_matrices
from .types import CandidateScore, RegistrationConfig, RegistrationResult, TransformCandidate


def _choose_best_candidate(
    source_poly: vtk.vtkPolyData,
    target_poly: vtk.vtkPolyData,
    candidates: list[TransformCandidate],
) -> tuple[TransformCandidate, list[CandidateScore]]:
    scored: list[tuple[TransformCandidate, CandidateScore]] = []
    for candidate in candidates:
        moved = apply_matrix_to_polydata(source_poly, candidate.matrix4x4)
        candidate.moved_source = moved
        score = score_candidate(candidate.name, moved, target_poly)
        scored.append((candidate, score))

    if not scored:
        raise RuntimeError("No valid PCA candidates available for scoring.")

    scored.sort(key=lambda x: x[1].symmetric_mean_distance)
    candidates_sorted = [x[0] for x in scored]
    scores_sorted = [x[1] for x in scored]
    return candidates_sorted[0], scores_sorted


def _run_icp_multistart(
    source_poly: vtk.vtkPolyData,
    target_poly: vtk.vtkPolyData,
    candidates_ranked: list[TransformCandidate],
    config: RegistrationConfig,
) -> tuple[TransformCandidate, Any, vtk.vtkPolyData]:
    best_item = None
    for candidate in candidates_ranked:
        moved = candidate.moved_source
        if moved is None:
            moved = apply_matrix_to_polydata(source_poly, candidate.matrix4x4)
        icp_result = run_icp(
            moved,
            target_poly,
            mode=config.icp_mode,
            max_iterations=config.icp_max_iterations,
            max_landmarks=config.icp_max_landmarks,
            max_mean_distance=config.icp_max_mean_distance,
        )
        final_matrix = compose_matrices(candidate.matrix4x4, icp_result.matrix4x4)
        registered = apply_matrix_to_polydata(source_poly, final_matrix)
        final_score = score_candidate(candidate.name, registered, target_poly)
        item = (candidate, icp_result, registered, final_matrix, final_score.symmetric_mean_distance)
        if best_item is None or item[-1] < best_item[-1]:
            best_item = item

    if best_item is None:
        raise RuntimeError("ICP multistart failed to produce a result.")
    return best_item[0], best_item[1], best_item[2]


def register_nii_segmentations(
    source_nii_path: str,
    target_nii_path: str,
    config: RegistrationConfig | None = None,
) -> RegistrationResult:
    config = config or RegistrationConfig()

    source_poly = load_binary_segmentation_nii_as_polydata(
        source_nii_path,
        label_value=config.source_label,
        smoothing_iterations=config.smoothing_iterations,
    )
    target_poly = load_binary_segmentation_nii_as_polydata(
        target_nii_path,
        label_value=config.target_label,
        smoothing_iterations=config.smoothing_iterations,
    )

    source_poly = clean_and_decimate(source_poly, config.decimation_reduction)
    target_poly = clean_and_decimate(target_poly, config.decimation_reduction)

    candidates, diagnostics = generate_pca_candidates(source_poly, target_poly)
    if not candidates:
        raise RuntimeError("PCA candidate generation returned zero valid transforms.")

    best_candidate, score_ranking = _choose_best_candidate(source_poly, target_poly, candidates)

    diagnostics["source_pca_unstable"] = pca_unstable(
        diagnostics["source_eigenvalues"], threshold=config.pca_unstable_threshold
    )
    diagnostics["target_pca_unstable"] = pca_unstable(
        diagnostics["target_eigenvalues"], threshold=config.pca_unstable_threshold
    )
    diagnostics["candidate_score_margin"] = (
        float(score_ranking[1].symmetric_mean_distance - score_ranking[0].symmetric_mean_distance)
        if len(score_ranking) > 1
        else None
    )

    source_is_unstable = bool(diagnostics["source_pca_unstable"])
    target_is_unstable = bool(diagnostics["target_pca_unstable"])

    if source_is_unstable or target_is_unstable:
        diagnostics["warning"] = (
            "PCA appears unstable from eigenvalue ratios; consider multi-start ICP selection."
        )

    if config.run_multistart_on_unstable_pca and (source_is_unstable or target_is_unstable):
        candidate_names = [s.candidate_name for s in score_ranking[: max(config.multistart_top_k, 1)]]
        subset = [c for c in candidates if c.name in set(candidate_names)]
        chosen_candidate, icp_result, registered_source = _run_icp_multistart(
            source_poly, target_poly, subset, config
        )
        pca_matrix = chosen_candidate.matrix4x4
    else:
        moved_best = best_candidate.moved_source
        if moved_best is None:
            moved_best = apply_matrix_to_polydata(source_poly, best_candidate.matrix4x4)
        icp_result = run_icp(
            moved_best,
            target_poly,
            mode=config.icp_mode,
            max_iterations=config.icp_max_iterations,
            max_landmarks=config.icp_max_landmarks,
            max_mean_distance=config.icp_max_mean_distance,
        )
        pca_matrix = best_candidate.matrix4x4
        registered_source = apply_matrix_to_polydata(
            source_poly,
            compose_matrices(pca_matrix, icp_result.matrix4x4),
        )
        chosen_candidate = best_candidate

    final_matrix = compose_matrices(pca_matrix, icp_result.matrix4x4)
    score_by_name = {s.candidate_name: s.symmetric_mean_distance for s in score_ranking}

    return RegistrationResult(
        source_polydata=source_poly,
        target_polydata=target_poly,
        registered_source_polydata=registered_source,
        pca_matrix=pca_matrix,
        icp_matrix=icp_result.matrix4x4,
        final_matrix=final_matrix,
        best_candidate_name=chosen_candidate.name,
        best_candidate_score=score_by_name.get(chosen_candidate.name, score_ranking[0].symmetric_mean_distance),
        candidate_scores=score_ranking,
        diagnostics=diagnostics,
    )
