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

### Registration (`final_matrix`)

During registration, the moving mask is the **source** and the fixed mask is the **target** (see `--geometry_mode` below). The JSON fields `pca_matrix`, `icp_matrix`, and `final_matrix` refer to that registration pair:

- `T_final = T_icp @ T_pca` (via `compose_matrices(T_pca, T_icp)`)

### Resampling (`resample_matrix`)

The report also stores **`resample_matrix`**: the world-space map used to move one mask into the other’s voxel grid for Dice and optional exports. Its meaning depends on `--geometry_mode`:

| `--geometry_mode` | Registration (moving → fixed) | `resample_matrix` | Typical use |
|-------------------|-------------------------------|-------------------|-------------|
| `us_to_ct` (default) | US → CT | US → CT | Evaluate US mask in CT space |
| `ct_in_us_inverse` | US → CT (then inverted for resampling) | CT → US | CT mask / CT volume in US space |
| `ct_in_us_direct` | CT → US | CT → US | CT mask / CT volume in US space |

Exported `.tfm` files use the same transform as `resample_matrix`, converted to ITK **LPS** for Slicer (`--save_tfm`). Filenames include `US_to_CT` or `CT_to_US` accordingly.

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

### Geometry modes (`--geometry_mode`)

- **`us_to_ct`** (default): register US → CT; Dice compares the US mask warped into CT voxel space against the CT mask. `resample_matrix` is **US → CT**.
- **`ct_in_us_inverse`**: register US → CT internally, then **invert** the transform for resampling; `resample_matrix` is **CT → US** (CT expressed in US space).
- **`ct_in_us_direct`**: register CT → US; `resample_matrix` is **CT → US**.

Use `ct_in_us_*` when you need transforms and masks aligned in **US** space. Example with transform export and a resampled moving mask:

```powershell
python -m pca_icp.run_pca_icp_registration --base_dir "C:\path\to\dataset_root" --out_dir "C:\path\to\output_reports" --geometry_mode ct_in_us_direct --icp_mode rigid --save_tfm --transform_dir "C:\path\to\output_reports\transforms" --save_resampled_moving_nii
```

### Main Options

- `--base_dir`: root folder containing `CT_masks` and `US_masks`.
- `--out_dir`: where JSON reports are written.
- `--geometry_mode`: `us_to_ct`, `ct_in_us_inverse`, or `ct_in_us_direct` (see above).
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
- `--save_tfm`: write `resample_matrix` as a `.tfm` for 3D Slicer (ITK LPS convention); filename suffix reflects `US_to_CT` or `CT_to_US`.
- `--transform_dir`: output folder for `.tfm` files (defaults to `out_dir`).
- `--save_resampled_moving_nii`: save the moving modality mask resampled onto the fixed grid (for `ct_in_us_*`, this is the **CT mask in US space**).

Each case writes a JSON report containing:

- `pca_matrix`, `icp_matrix`, `final_matrix` (registration pair)
- `resample_matrix`, `resample_transform_name` (`US_to_CT` or `CT_to_US`), `geometry_mode`
- paths used for registration vs resampling (`registration_*`, `resample_*`)
- Dice score for the resampled-vs-fixed mask comparison
- `final_transform_tfm_path` when `--save_tfm` is set
- `resampled_moving_nii_path` when `--save_resampled_moving_nii` is set
- selected candidate and candidate scores
- PCA diagnostics and instability flags

### Apply existing transform: resample CT into US space

When you already have a **CT → US** transform from `ct_in_us_*` runs (`*_CT_to_US_pca_icp.tfm` or `resample_matrix` in the JSON), you can resample a **full CT volume** onto the US reference grid with linear interpolation (requires **SimpleITK**, same as `--save_tfm`).

Use the same US image or mask that defines the US registration target as `--us_reference` so the output grid matches the pipeline.

```powershell
python -m pca_icp.resample_ct_to_us --ct "C:\path\to\314L_imgCT.nii.gz" --us_reference "C:\path\to\dataset_root\US_masks\314L_maskUS.nii.gz" --tfm "C:\path\to\transforms\314L_CT_to_US_pca_icp.tfm" --out "C:\path\to\314L_CT_in_US_space.nii.gz"
```

Using the JSON report instead of `.tfm`:

```powershell
python -m pca_icp.resample_ct_to_us --ct "C:\path\to\314L_imgCT.nii.gz" --us_reference "C:\path\to\dataset_root\US_masks\314L_maskUS.nii.gz" --report_json "C:\path\to\output_reports\314L_pca_icp_report.json" --out "C:\path\to\314L_CT_in_US_space.nii.gz"
```

Optional: `--default_pixel_value` (default `-1000`) sets the intensity outside the sampled CT field (typical HU for air).

If the resampled CT appears misaligned, confirm `--us_reference` is the same US NIfTI used as the registration target for that case, and that the transform is **CT → US** (`CT_to_US` in the filename or `resample_transform_name` in JSON).

### Batch mode (TRUSTED-style folders)

When data are laid out like `kidney_dataset/TRUSTED/` with `CT_images/`, `CT_masks/`, `US_masks/`, and `init_transf/` (one CT→US `.tfm` per case), you can resample **all** CT volumes and CT masks onto each case’s US mask grid in one run. Case IDs are parsed the same way as the PCA+ICP dataset indexer (leading `314L_`-style prefix).

- **Transforms**: default filename per case is `{case_id}_CT_to_US_pca_icp.tfm` under `init_transf/`; override with `--tfm_basename_template` (must include `{case_id}`).
- **Outputs**: `out_dir/resampled_CT_images/` and `out_dir/resampled_CT_masks/`; filenames add `_in_US_space` before the NIfTI suffix. By default the suffix is **`.nii` (uncompressed)** so very large volumes do not hit ITK `** ERROR: NWAD: wrote only 0 of N bytes` / empty files on tight disk quota or network filesystems. For gzip, use `--batch_nifti_suffix .nii.gz` and `--batch_nifti_gzip` (can still fail on multi-GB writes if space is low).
- **`--strict`**: exit with an error if any case is skipped (e.g. missing `.tfm`) or any resampling throws.

```powershell
python -m pca_icp.resample_ct_to_us `
  --batch_root "C:\path\to\kidney_dataset\TRUSTED" `
  --out_dir "C:\path\to\TRUSTED_resampled"
```

Optional: `--ct_images_subdir`, `--ct_masks_subdir`, `--us_masks_subdir`, `--init_transf_subdir` if your subfolder names differ.

Gzip batch outputs (not default):

```powershell
python -m pca_icp.resample_ct_to_us --batch_root "C:\path\to\kidney_dataset\TRUSTED" --out_dir "C:\path\to\TRUSTED_resampled" --batch_nifti_suffix .nii.gz --batch_nifti_gzip
```

### Crop CT images/masks to fixed XY (keep all Z)

If you want in-plane normalization (for example `512x512`) but do **not** want to lose kidney anatomy, use the mask-aware crop utility. It:

- builds the crop center from each case CT mask bounding box;
- keeps all Z slices unchanged;
- pads if source XY is smaller than target;
- validates that the cropped mask keeps the same positive-voxel count as the original.

```powershell
python -m pca_icp.crop_ct_by_mask `
  --batch_root "C:\path\to\kidney_dataset\TRUSTED" `
  --out_dir "C:\path\to\TRUSTED_cropped" `
  --target_xy 512 512 `
  --bbox_margin_xy 12 12
```

Outputs:

- `cropped_CT_images/`
- `cropped_CT_masks/`
- `crop_reports/` (per-case JSON with bbox, crop origin, and voxel-retention checks)

Useful options:

- `--ct_pad_value -1000` (default): value used when padding CT.
- `--strict`: stop immediately on first failing case.
- `--nifti_suffix .nii` for uncompressed outputs, or `.nii.gz` + `--nifti_gzip` for gzip.

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
