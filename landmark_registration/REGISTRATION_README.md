# Landmark Registration Pipeline (MRI->US, JSON Manifest)

This mode runs landmark-based registration from moving landmarks to fixed landmarks using MNI `.tag` files.

Default modality convention:
- moving: `MRI`
- fixed: `US`
- transform direction: `MRI -> US`

It is batch-oriented and dataset-agnostic through a JSON manifest.

## Input Landmark Format

MNI `.tag` files are expected to contain 2-volume points in this order per row:

- `moving_x moving_y moving_z fixed_x fixed_y fixed_z ""`

For RESECT this maps naturally to:
- moving = MRI landmark
- fixed = US landmark

## Manifest JSON Format

Run from repository root and provide a JSON manifest with a `cases` array.

### Required per-case keys

- `case_id`
- `tag_path`
- `moving_image_path`
- `fixed_image_path`

`moving_image_path` and `fixed_image_path` are required keys but values can be empty/null when LC2 is not computable. LC2 is computed only when both paths are valid readable files.

### Example manifest

```json
{
  "cases": [
    {
      "case_id": "Case1",
      "tag_path": "C:/path/to/RESECT-Seg/Case1/Landmarks/Case1-MRI-beforeUS.tag",
      "moving_image_path": "C:/path/to/RESECT-Seg/Case1/MRI/Case1-MRI.nii.gz",
      "fixed_image_path": "C:/path/to/RESECT-Seg/Case1/US/Case1-US.nii.gz"
    }
  ]
}
```

## How To Create The Manifest

### Option A: Create manually

1. Create a file such as `C:/path/to/resect_manifest.json`.
2. Paste the template above.
3. Add one object per case in `cases`.
4. Save and run the batch CLI.

### Option B: Generate with Python script

```python
import json
from pathlib import Path

root = Path(r"C:\path\to\RESECT-Seg")
cases = []

for case_dir in sorted(root.glob("Case*")):
    case_id = case_dir.name
    cases.append(
        {
            "case_id": case_id,
            "tag_path": str(case_dir / "Landmarks" / f"{case_id}-MRI-beforeUS.tag"),
            "moving_image_path": str(case_dir / "MRI" / f"{case_id}-MRI.nii.gz"),
            "fixed_image_path": str(case_dir / "US" / f"{case_id}-US.nii.gz"),
        }
    )

manifest = {"cases": cases}
Path("resect_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print("Wrote resect_manifest.json")
```

Adjust naming/path logic to your local dataset layout.

### Option C: Generate with the built-in helper script (recommended)

Use the provided script to auto-detect RESECT case inputs (`Landmarks`, `MRI` FLAIR, `US-before`)
and write a valid manifest:

```powershell
python -m landmark_registration.create_resect_manifest --dataset_root "C:\path\to\RESECT-Data"
```

Optional flags:
- `--out_manifest "C:\path\to\custom_manifest.json"` to choose output file location.
- `--allow_partial` to write only valid cases when some cases fail validation.

## CLI

Activate your environment, then run from repository root:

```powershell
python -m landmark_registration.run_landmark_registration --manifest "C:\path\to\resect_manifest.json" --out_dir "C:\path\to\landmark_outputs"
```

### Common run modes

Rigid transform (recommended default):

```powershell
python -m landmark_registration.run_landmark_registration --manifest "C:\path\to\resect_manifest.json" --out_dir "C:\path\to\landmark_outputs_rigid" --model rigid
```

Similarity transform:

```powershell
python -m landmark_registration.run_landmark_registration --manifest "C:\path\to\resect_manifest.json" --out_dir "C:\path\to\landmark_outputs_similarity" --model similarity
```

Affine transform:

```powershell
python -m landmark_registration.run_landmark_registration --manifest "C:\path\to\resect_manifest.json" --out_dir "C:\path\to\landmark_outputs_affine" --model affine
```

Keep per-point TRE errors in case reports:

```powershell
python -m landmark_registration.run_landmark_registration --manifest "C:\path\to\resect_manifest.json" --out_dir "C:\path\to\landmark_outputs" --save_per_point_errors
```

Override manifest keys for other datasets:

```powershell
python -m landmark_registration.run_landmark_registration --manifest "C:\path\to\custom_manifest.json" --out_dir "C:\path\to\out" --manifest_cases_key "items" --manifest_case_id_key "id" --manifest_tag_key "landmark_file" --manifest_moving_img_key "moving_img" --manifest_fixed_img_key "fixed_img"
```

## Main Options

- `--manifest`: path to JSON manifest.
- `--out_dir`: output root for matrices and reports.
- `--model`: `rigid`, `similarity`, or `affine`.
- `--moving_modality`: label used in outputs (default `MRI`).
- `--fixed_modality`: label used in outputs (default `US`).
- `--manifest_cases_key`: manifest array key (default `cases`).
- `--manifest_case_id_key`: case id key (default `case_id`).
- `--manifest_tag_key`: landmark file key (default `tag_path`).
- `--manifest_moving_img_key`: moving image key (default `moving_image_path`).
- `--manifest_fixed_img_key`: fixed image key (default `fixed_image_path`).
- `--save_per_point_errors`: include per-point TRE vectors in case JSON.

## Outputs

For each run:
- `out_dir/matrices/<case_id>_<moving>_to_<fixed>_landmark_<model>.txt` (4x4 transform matrix)
- `out_dir/matrices/<case_id>_<moving>_to_<fixed>_landmark_<model>.tfm` (ITK/SimpleITK affine transform)
- `out_dir/reports/<case_id>_landmark_report.json` (case report)
- `out_dir/landmark_registration_summary.json` (run summary)

Case report includes:
- transform direction and model,
- input paths,
- transform paths (`transform_matrix_txt_path`, `transform_matrix_tfm_path`),
- TRE before/after and deltas,
- LC2 status and values (before/after/delta when available),
- warnings for skipped/failed LC2 cases.

## Resample MRI Into US Space (Post-registration)

After running landmark registration, you can resample MRI volumes into the fixed US grid using the saved
per-case reports (`transform_matrix_4x4` / moving->fixed convention).

All successful cases from a run summary:

```powershell
python -m landmark_registration.resample_mri_to_us --summary_json "C:\path\to\landmark_outputs\landmark_registration_summary.json" --out_dir "C:\path\to\landmark_outputs\resampled_mri_in_us"
```

Single case report:

```powershell
python -m landmark_registration.resample_mri_to_us --report_json "C:\path\to\landmark_outputs\reports\Case1_landmark_report.json" --out_dir "C:\path\to\landmark_outputs\resampled_mri_in_us"
```

Optional controls:
- `--case_id Case1` (with `--summary_json`) to process one case from a summary.
- `--interpolation nearest` for label-like volumes (default is `linear`).
- `--out_suffix .nii` to write uncompressed outputs (`.nii.gz` by default).

## Transform Convention

All 4x4 matrices are homogeneous transforms using column-vector convention:

- `p_fixed_h = T_moving_to_fixed @ p_moving_h`

This mode computes and saves moving->fixed transforms.

## Notes

- At least 3 landmark pairs are required per case.
- LC2 is optional by design; if image geometry or paths are invalid, LC2 is skipped and recorded in report warnings.
- Existing repository documentation may show `utils/...` script paths in affine docs; for this mode use module execution:
  - `python -m landmark_registration.run_landmark_registration`
