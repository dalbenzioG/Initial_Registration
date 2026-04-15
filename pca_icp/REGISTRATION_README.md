# PCA + ICP Kidney Registration (VTK)

This package runs a rigid global-to-local registration of US kidney segmentation to CT kidney segmentation:

1. NIfTI binary segmentation -> surface mesh (`vtkPolyData`)
2. PCA coarse alignment with 4 right-handed sign candidates
3. Candidate scoring (symmetric mean distance + Hausdorff95)
4. ICP refinement (rigid/similarity/affine mode)

## Data Layout

The dataset indexer expects:

- `CT_masks/*.nii or *.nii.gz`
- `US_masks/*.nii or *.nii.gz`

Case IDs are parsed with pattern `(\d+[LR])_`, e.g. `314L_maskCT.nii.gz`.

## Transform Convention

All 4x4 matrices use:

- column-vector homogeneous points
- `p_out_h = T @ p_in_h`
- source is US, target is CT
- final transform maps US -> CT

Composition order:

- `T_final = T_icp @ T_pca`
- implemented via `compose_matrices(T_pca, T_icp)`

## CLI

Activate your environment, then run from repository root:

```powershell
.\pcarigid\Scripts\Activate.ps1
python -m pca_icp.run_pca_icp_registration --base_dir "C:\path\to\dataset_root" --out_dir "C:\path\to\output_reports"
```

### Common Run Modes

Default (recommended start, rigid ICP):

```powershell
python -m pca_icp.run_pca_icp_registration --base_dir "C:\path\to\dataset_root" --out_dir "C:\path\to\output_reports" --icp_mode rigid
```

Fast debug run (lighter mesh + fewer ICP iterations):

```powershell
python -m pca_icp.run_pca_icp_registration --base_dir "C:\path\to\dataset_root" --out_dir "C:\path\to\output_reports_debug" --icp_mode rigid --smoothing_iterations 3 --decimation_reduction 0.8 --icp_max_iterations 30 --icp_max_landmarks 500
```

More robust run for difficult cases (more ICP iterations + more multistart):

```powershell
python -m pca_icp.run_pca_icp_registration --base_dir "C:\path\to\dataset_root" --out_dir "C:\path\to\output_reports_robust" --icp_mode rigid --icp_max_iterations 150 --multistart_top_k 3 --pca_unstable_threshold 1.15
```

Save final transforms as `.tfm` for 3D Slicer:

```powershell
python -m pca_icp.run_pca_icp_registration --base_dir "C:\path\to\dataset_root" --out_dir "C:\path\to\output_reports" --icp_mode rigid --save_tfm --transform_dir "C:\path\to\output_reports\transforms"
```

Try similarity or affine ICP (only if justified by your data):

```powershell
python -m pca_icp.run_pca_icp_registration --base_dir "C:\path\to\dataset_root" --out_dir "C:\path\to\output_reports_similarity" --icp_mode similarity
python -m pca_icp.run_pca_icp_registration --base_dir "C:\path\to\dataset_root" --out_dir "C:\path\to\output_reports_affine" --icp_mode affine
```

### Main Options

- `--base_dir`: root folder containing `CT_masks` and `US_masks`.
- `--out_dir`: where JSON reports are written.
- `--source_label`: US label value (default `1`).
- `--target_label`: CT label value (default `1`).
- `--smoothing_iterations`: mesh smoothing strength before registration.
- `--decimation_reduction`: mesh reduction ratio (`0.0` keeps full mesh).
- `--icp_mode`: `rigid` (default), `similarity`, or `affine`.
- `--icp_max_iterations`: max ICP iterations.
- `--icp_max_landmarks`: max sampled landmarks used by ICP.
- `--icp_max_mean_distance`: ICP stop threshold.
- `--pca_unstable_threshold`: eigenvalue-ratio threshold for PCA ambiguity.
- `--multistart_top_k`: number of top PCA candidates to retry in unstable cases.
- `--save_tfm`: export final US->CT transform as `.tfm`.
- `--transform_dir`: output folder for `.tfm` files (defaults to `out_dir`).

Each case writes a JSON report containing:

- PCA matrix
- ICP matrix
- final matrix
- Dice score (US mask transformed into CT grid vs CT mask)
- selected candidate and candidate scores
- PCA diagnostics and instability flags
- optional `.tfm` path when `--save_tfm` is enabled

## Notes

- ICP is local; PCA selection quality strongly affects convergence.
- If PCA is unstable (near-symmetric eigenvalues), the pipeline can run multi-start ICP on top candidates.
- Affine mode is available but not the safe default for anatomy-driven registration.

## Troubleshooting

`ImportError: cannot import name 'GenericAlias' from 'types'`

- Cause: running the script path directly can shadow stdlib `types` because of `pca_icp/types.py`.
- Fix: run as a module from repo root:

```powershell
python -m pca_icp.run_pca_icp_registration --base_dir "C:\path\to\data" --out_dir "C:\path\to\out"
```

`ImportError: DLL load failed while importing vtk... (Application Control policy blocked this file)`

- Cause: Windows policy/AppLocker blocking one or more VTK binaries.
- Fixes:
  - use Slicer Python for execution, or
  - request IT allowlist for `vtkmodules` binaries in the environment, or
  - recreate clean environment and retry.

`TypeError: Object of type bool is not JSON serializable`

- Cause: NumPy scalar types in diagnostics (`np.bool_`, etc.) during JSON export.
- Fix: implemented in runner via recursive conversion to native Python types.
