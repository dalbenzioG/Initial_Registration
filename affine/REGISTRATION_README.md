# CT-US Kidney Registration Pipeline (TRUSTED)

Modular SimpleITK-based pipeline for CT↔US kidney registration: metadata validation, SDM generation, and multi-stage affine registration.

## Design Choices

- **Utility-oriented layout (MIR-style)**: Core logic lives in `registration_utils.py`; `validate_and_generate_sdm.py` and `run_affine_registration.py` are thin CLI scripts that call these helpers. Reusable functions avoid duplicated pairing and transform logic.
- **Metadata validation first**: Mask/image origin, spacing, and direction are checked explicitly. `--fail_on_mismatch` exits on any mismatch; `--fix_metadata` copies image metadata into masks only when explicitly requested.
- **SDMs for registration**: Signed distance maps provide a smooth metric landscape for gradient-based optimization; MeanSquares on SDMs works better than direct mask overlap.
- **Multi-stage registration**: MOMENTS initialization → rigid (Euler3D) → affine, with multi-resolution (shrink factors, smoothing sigmas). `SetOptimizerScalesFromPhysicalShift()` ensures commensurate parameter scaling.
- **Pathlib throughout**: Paths handled via `pathlib.Path`; `find_files_by_suffixes` centralizes filename parsing and pairing.

## Transform Direction and File Naming

- **tx_us_to_ct**: Maps US physical coordinates → CT physical coordinates. Use this to resample US into CT space.
- **tx_ct_to_us**: Inverse of above; maps CT → US.
- Saved transforms use ITK "from parent" convention for Slicer compatibility: `US_to_CT_affine.tfm` stores CT→US (put US under transform); `CT_to_US_affine.tfm` stores US→CT (put CT under transform).
- Aligned output: `{key}_US_aligned_to_CT.nii.gz` (US resampled into CT geometry).
- SDM filenames: `{key}_imgCT_sdm.nii.gz`, `{key}_imgUS_sdm.nii.gz` (e.g. `200L_imgCT_sdm.nii.gz`).

## Example CLI Commands

Run from the repository root.

### Step 1: Validate and generate SDMs

```bash
python utils/validate_and_generate_sdm.py \
  --ct_img_dir /path/to/TRUSTED/CT_DATA/CT_images \
  --us_img_dir /path/to/TRUSTED/US_DATA/US_images \
  --ct_mask_dir /path/to/TRUSTED/CT_DATA/CT_masks \
  --us_mask_dir /path/to/TRUSTED/US_DATA/US_masks \
  --out_dir /path/to/TRUSTED/sdms \
  --report_format json
```

With metadata repair (use with caution):

```bash
python utils/validate_and_generate_sdm.py \
  --ct_img_dir ... --us_img_dir ... --ct_mask_dir ... --us_mask_dir ... \
  --out_dir /path/to/sdms \
  --fix_metadata
```

Strict mode (exit on any metadata mismatch):

```bash
python utils/validate_and_generate_sdm.py \
  ... \
  --fail_on_mismatch
```

Resume interrupted run (skip pairs whose SDMs already exist):

```bash
python utils/validate_and_generate_sdm.py \
  ... same args as before ... \
  --out_dir /path/to/sdms \
  --skip_existing
```

### Step 2: Run registration

```bash
python utils/run_affine_registration.py \
  --sdm_dir /path/to/TRUSTED/sdms \
  --ct_img_dir /path/to/TRUSTED/CT_DATA/CT_images \
  --us_img_dir /path/to/TRUSTED/US_DATA/US_images \
  --out_dir /path/to/TRUSTED/aligned \
  --save_transform \
  --transform_dir /path/to/TRUSTED/transforms
```

Transforms only (no aligned images, saves both US→CT and CT→US):

```bash
python utils/run_affine_registration.py \
  --sdm_dir /path/to/TRUSTED/sdms \
  --ct_img_dir /path/to/TRUSTED/CT_DATA/CT_images \
  --us_img_dir /path/to/TRUSTED/US_DATA/US_images \
  --transform_dir /path/to/TRUSTED/transforms \
  --transforms_only --save_transform
```

With custom optimizer settings:

```bash
python utils/run_affine_registration.py \
  --sdm_dir ... --ct_img_dir ... --us_img_dir ... --out_dir ... \
  --rigid_iterations 300 \
  --affine_iterations 500 \
  --shrink_factors "8,4,2,1" \
  --smoothing_sigmas "4.0,2.0,1.0,0.0"
```

## Assumptions

- TRUSTED naming: `{patient}{side}_seg.nii.gz`, `{patient}{side}_maskUS.nii.gz`, `{patient}{side}_imgCT.nii.gz`, `{patient}{side}_imgUS.nii.gz`.
- Kidney key format: `{patient}{side}` (e.g. `200L`, `200R`).
- Masks and images are in physical space; metadata (origin, spacing, direction) must match for valid registration.
- SimpleITK only; no Elastix.
- Python 3.10+.
