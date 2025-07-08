import os
import logging
import SimpleITK as sitk
import pandas as pd
from registration_utils import compute_quick_metric, register_to_atlas, log_progress
from tqdm import tqdm
import nibabel as nib
import numpy as np
from skimage.transform import resize
from volume_utils import process_ct_and_crop_abdomen


def get_nifti_dimensions(file_path):
    """Load NIfTI file and return its data, dimensions, and header."""
    img = nib.load(file_path)
    data = img.get_fdata()
    return data, data.shape, img.header

def find_smallest_dimensions(file_paths):
    """Find the smallest dimensions across all volumes."""
    dimensions = []
    for file_path in file_paths:
        _, shape, _ = get_nifti_dimensions(file_path)
        dimensions.append(shape)
    return tuple(np.min(dimensions, axis=0))


def rescale_volume(data, target_shape):
    """Rescale volume to target shape using interpolation."""
    zoom_factors = [t / s for t, s in zip(target_shape, data.shape)]
    rescaled = resize(data, target_shape, order=1, anti_aliasing=True, preserve_range=True)
    return rescaled

def pad_to_shape(data, target_shape):
    """Pad volume to target shape with minimal padding."""
    current_shape = data.shape
    padding = [(0, max(0, t - c)) for t, c in zip(target_shape, current_shape)]
    padded = np.pad(data, padding, mode='constant', constant_values=0)
    return padded, padding

def load_phase_labels(labels_path):
    """Load phase labels from CSV file."""
    log_progress(f"Loading phase labels from {labels_path}...")
    df = pd.read_csv(labels_path)
    # Create a dictionary mapping series_id to phase
    phase_map = {}
    for _, row in df.iterrows():
        series_id = row['SeriesInstanceUID']
        phase = row['Label']
        phase_map[series_id] = phase.lower()
    log_progress(f"✓ Loaded {len(phase_map)} phase labels")
    return phase_map

def load_images_from_directory(input_dir):
    """
    Load all NIfTI volumes from the input directory where files are named by series_id.nii
    """
    log_progress(f"Scanning directory for NIfTI files: {input_dir}")
    volume_paths = {}
    for file in os.listdir(input_dir):
        if file.endswith(".nii") or file.endswith(".nii.gz"):
            full_path = os.path.join(input_dir, file)
            series_id = os.path.splitext(file)[0].replace(".nii", "").replace(".gz", "")
            volume_paths[series_id] = full_path
    
    log_progress(f"✓ Found {len(volume_paths)} NIfTI files")
    return volume_paths

def select_atlas(non_contrast_paths_dict):
    """
    Select the most representative volume (atlas) using average similarity.
    Only considers non-contrast volumes as atlas candidates.
    non_contrast_paths_dict: {key: file_path}
    """
    volume_keys = list(non_contrast_paths_dict.keys())
    log_progress(f"Selecting atlas from {len(volume_keys)} non-contrast volumes...")
    metrics = {}
    for i in tqdm(range(len(volume_keys)), desc="Computing atlas metrics"):
        ref_key = volume_keys[i]
        ref_path = non_contrast_paths_dict[ref_key]
        ref_img = sitk.ReadImage(ref_path)
        total_metric = 0
        for j in range(len(volume_keys)):
            if i == j:
                continue
            cmp_key = volume_keys[j]
            cmp_path = non_contrast_paths_dict[cmp_key]
            cmp_img = sitk.ReadImage(cmp_path)
            metric = compute_quick_metric(ref_img, cmp_img, metric_type='ncc')
            total_metric += metric
            del cmp_img
        metrics[ref_key] = total_metric
        del ref_img
    best_key = min(metrics, key=metrics.get)
    log_progress(f"✓ Selected atlas: {best_key}")
    return best_key, non_contrast_paths_dict[best_key]

def main(input_dir, output_dir, labels_path):
    os.makedirs(output_dir, exist_ok=True)
    log_progress(f"Starting registration pipeline...")
    log_progress(f"Input directory: {input_dir}")
    log_progress(f"Output directory: {output_dir}")

    # Load phase labels and find all volume paths
    phase_map = load_phase_labels(labels_path)
    volume_paths = load_images_from_directory(input_dir)

    # Build a dictionary of {key: (file_path, phase)}
    volume_info = {key: (path, phase_map.get(key, 'unknown')) for key, path in volume_paths.items()}

    # For atlas selection, get only non-contrast paths
    non_contrast_paths_dict = {key: path for key, (path, phase) in volume_info.items() if phase == 'non-contrast'}
    if not non_contrast_paths_dict:
        raise ValueError("No valid non-contrast volumes could be loaded")

    # Select atlas
    atlas_key, atlas_path = select_atlas(non_contrast_paths_dict)
    atlas_img = sitk.ReadImage(atlas_path)

    # Process each volume by loading one at a time
    log_progress("Starting registration of volumes to atlas...")
    for key, (vol_path, phase) in tqdm(volume_info.items(), desc="Registering volumes"):
        try:
            log_progress(f"Processing volume {key} (phase: {phase})...")
            # Load atlas and moving volume
            moving_img = sitk.ReadImage(vol_path)
            # Register to atlas
            transform, registered_img = register_to_atlas(atlas_img, moving_img, is_atlas=(key == atlas_key), phase=phase, debug_dir="utils/debug/ncct_cect/vindr_ds/debug_registeration", key=key)
            # Convert back to NIfTI and save
            # registered_np = sitk.GetArrayFromImage(registered_img)
            # For affine, try to get affine from the original NIfTI file
            out_path = os.path.join(output_dir, f"{key}_registered.nii.gz")
            sitk.WriteImage(registered_img, out_path)
            log_progress(f"✓ Saved registered image to {out_path}")
            # Clear memory
            del moving_img
            del registered_img
            
        except Exception as e:
            log_progress(f"Failed to process volume {key}: {e}", level='error')
            continue
    log_progress("✓ Registration pipeline completed successfully")

if __name__ == "__main__":
    INPUT_DIR = "utils/debug/ncct_cect/vindr_ds/padded_volumes"
    OUTPUT_DIR = "utils/debug/ncct_cect/vindr_ds/registered_volumes"
    LABELS_PATH = "utils/debug/ncct_cect/vindr_ds/labels.csv"
    main(INPUT_DIR, OUTPUT_DIR, LABELS_PATH)