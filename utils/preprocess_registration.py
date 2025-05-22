import os
import json
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm
from pathlib import Path
import pandas as pd
from utils.archived_volume_utils import resample_spacing
from registration_utils import register_to_atlas

# --- CONFIG ---
ALIGNED_SLICES_DIR = '../../ncct_cect/vindr_ds/aligned_slices'
REGISTERED_DIR = '../../ncct_cect/vindr_ds/registered_volumes'
ATLAS_SAVE_PATH = '../../ncct_cect/vindr_ds/atlas/reference.nii.gz'
LABELS_CSV = '../../ncct_cect/vindr_ds/labels.csv'

os.makedirs(REGISTERED_DIR, exist_ok=True)

# --- STEP 1: Load label info ---
labels_df = pd.read_csv(LABELS_CSV)
labels_df = labels_df.dropna(subset=["StudyInstanceUID", "SeriesInstanceUID", "Phase"])
phase_labels = {
    (row["StudyInstanceUID"], row["SeriesInstanceUID"]): row["Phase"].lower()
    for _, row in labels_df.iterrows()
}

# --- STEP 2: Load All Aligned Volumes ---
print("Loading cropped volumes for registration...")
volume_paths = sorted(list(Path(ALIGNED_SLICES_DIR).rglob("*.nii.gz")))
all_volumes = []
metadata = []

for p in volume_paths:
    name = p.stem
    try:
        study_uid, series_uid = name.split("_")[:2]
    except ValueError:
        print(f"Skipping malformed filename: {name}")
        continue
    phase = phase_labels.get((study_uid, series_uid), None)
    if phase is None:
        print(f"Skipping {name} (no label info)")
        continue
    img = sitk.ReadImage(str(p))
    all_volumes.append(img)
    metadata.append((p, study_uid, series_uid, phase))

# --- STEP 3: Select Best Atlas from Non-Contrast ---
def compute_affine_distance(fixed, moving):
    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(50)
    registration.SetOptimizerAsRegularStepGradientDescent(1.0, 1e-6, 50)
    registration.SetInterpolator(sitk.sitkLinear)
    initial_transform = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )
    registration.SetInitialTransform(initial_transform)
    final_transform = registration.Execute(fixed, moving)
    params = np.array(final_transform.GetParameters())
    return np.linalg.norm(params)

non_contrast_indices = [i for i, (_, _, _, phase) in enumerate(metadata) if "non" in phase]
if not non_contrast_indices:
    raise ValueError("No non-contrast cases found for atlas selection.")

print("Selecting atlas from non-contrast images...")
errors = []
for i in non_contrast_indices:
    fixed = all_volumes[i]
    diffs = []
    for j in non_contrast_indices:
        if i == j:
            continue
        moving = all_volumes[j]
        diffs.append(compute_affine_distance(fixed, moving))
    errors.append(np.mean(diffs))

best_idx = non_contrast_indices[np.argmin(errors)]
atlas_image = all_volumes[best_idx]
atlas_path = metadata[best_idx][0]
print(f"Selected atlas: {atlas_path.name}")
sitk.WriteImage(atlas_image, ATLAS_SAVE_PATH)

# --- STEP 4: Register All Volumes to Atlas ---
for i, (vol_path, study_uid, series_uid, phase) in enumerate(tqdm(metadata)):
    moving_image = all_volumes[i]
    transform = register_to_atlas(fixed=atlas_image, moving=moving_image)
    registered_image = sitk.Resample(moving_image, atlas_image, transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())

    # Save registered image
    filename = vol_path.stem + "_registered.nii.gz"
    save_path = os.path.join(REGISTERED_DIR, filename)
    sitk.WriteImage(registered_image, save_path)

print("Registration complete. All images saved to:", REGISTERED_DIR)
