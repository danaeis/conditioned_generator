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

def select_atlas(volumes, phase_map):
    """
    Select the most representative volume (atlas) using average similarity.
    Only considers non-contrast volumes as atlas candidates.
    """
    # Filter for non-contrast volumes only
    non_contrast_volumes = {
        key: img for key, img in volumes.items() 
        if phase_map.get(key, "").lower() == "non-contrast"
    }
    
    if not non_contrast_volumes:
        raise ValueError("No non-contrast volumes found for atlas selection")
        
    volume_keys = list(non_contrast_volumes.keys())
    log_progress(f"Selecting atlas from {len(volume_keys)} non-contrast volumes...")
    
    metrics = {}
    for i in tqdm(range(len(volume_keys)), desc="Computing atlas metrics"):
        ref_key = volume_keys[i]
        ref_img = non_contrast_volumes[ref_key]
        total_metric = 0
        for j in range(len(volume_keys)):
            if i == j:
                continue
            cmp_img = non_contrast_volumes[volume_keys[j]]
            metric = compute_quick_metric(ref_img, cmp_img, metric_type='ncc')
            total_metric += metric
        metrics[ref_key] = total_metric
    
    best_key = min(metrics, key=metrics.get)
    log_progress(f"✓ Selected atlas: {best_key}")
    return best_key, non_contrast_volumes[best_key]

def main(input_dir, output_dir, labels_path):
    os.makedirs(output_dir, exist_ok=True)
    log_progress(f"Starting registration pipeline...")
    log_progress(f"Input directory: {input_dir}")
    log_progress(f"Output directory: {output_dir}")

    # Load phase labels and find all volume paths
    phase_map = load_phase_labels(labels_path)
    volume_paths = load_images_from_directory(input_dir)
    
    # First pass: Process all volumes to remove CT table while maintaining dimensions
    processed_volumes = {}
    for key, path in tqdm(volume_paths.items(), desc="Preprocessing volumes"):
        try:
            # Load volume
            img = nib.load(path)
            data = img.get_fdata()
            
            # Process volume (remove table, focus on abdomen) while maintaining dimensions
            processed_data, processed_img = process_ct_and_crop_abdomen(data, img.affine)
            processed_volumes[key] = processed_img
            
        except Exception as e:
            log_progress(f"Failed to process volume {key}: {e}", level='error')
            continue
    
    # Select atlas from non-contrast volumes
    log_progress("Selecting atlas from non-contrast volumes...")
    non_contrast_volumes = {
        key: sitk.GetImageFromArray(vol.get_fdata()) 
        for key, vol in processed_volumes.items() 
        if phase_map.get(key, "").lower() == "non-contrast"
    }
    
    if not non_contrast_volumes:
        raise ValueError("No valid non-contrast volumes could be loaded")
    
    # Select atlas
    atlas_key, atlas_img = select_atlas(non_contrast_volumes, phase_map)
    
    # Clear non-contrast volumes from memory
    non_contrast_volumes.clear()
    
    # Process each volume by loading one at a time
    log_progress("Starting registration of volumes to atlas...")
    for key, vol in tqdm(processed_volumes.items(), desc="Registering volumes"):
        try:
            log_progress(f"Processing volume {key} (phase: {phase_map.get(key, 'unknown')})...")
            
            # Convert to SimpleITK for registration
            moving_img = sitk.GetImageFromArray(vol.get_fdata())
            moving_img.CopyInformation(sitk.GetImageFromArray(processed_volumes[atlas_key].get_fdata()))
            
            # Register to atlas
            transform, registered_img = register_to_atlas(atlas_img, moving_img, is_atlas=(key == atlas_key))
            
            # Convert back to NIfTI and save
            registered_np = sitk.GetArrayFromImage(registered_img)
            registered_nifti = nib.Nifti1Image(registered_np, vol.affine, header=vol.header)
            
            out_path = os.path.join(output_dir, f"{key}_registered.nii.gz")
            nib.save(registered_nifti, out_path)
            log_progress(f"✓ Saved registered image to {out_path}")
            
            # Clear memory
            del moving_img
            del registered_img
            
        except Exception as e:
            log_progress(f"Failed to process volume {key}: {e}", level='error')
            continue
    
    log_progress("✓ Registration pipeline completed successfully")

if __name__ == "__main__":
    INPUT_DIR = "utils/debug/ncct_cect/vindr_ds/cropped_volumes"
    OUTPUT_DIR = "utils/debug/ncct_cect/vindr_ds/registered_volumes"
    LABELS_PATH = "utils/debug/ncct_cect/vindr_ds/labels.csv"
    main(INPUT_DIR, OUTPUT_DIR, LABELS_PATH)