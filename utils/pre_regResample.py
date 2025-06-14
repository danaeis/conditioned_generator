import os
import logging
import SimpleITK as sitk
import pandas as pd
from registration_utils import compute_quick_metric, register_to_atlas, log_progress
from tqdm import tqdm

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

    # Step 1: Load phase labels and find all volume paths
    phase_map = load_phase_labels(labels_path)
    volume_paths = load_images_from_directory(input_dir)
    
    # Step 2: Select atlas by loading only non-contrast volumes
    log_progress("Loading non-contrast volumes for atlas selection...")
    non_contrast_volumes = {}
    for key, path in tqdm(volume_paths.items(), desc="Loading non-contrast volumes"):
        if phase_map.get(key, "").lower() == "non-contrast":
            try:
                non_contrast_volumes[key] = sitk.ReadImage(path)
            except Exception as e:
                log_progress(f"Failed to load non-contrast volume {key}: {e}", level='warning')
                continue
    
    if not non_contrast_volumes:
        raise ValueError("No valid non-contrast volumes could be loaded")
    
    # Select atlas from non-contrast volumes
    atlas_key, atlas_img = select_atlas(non_contrast_volumes, phase_map)
    
    # Clear non-contrast volumes from memory
    non_contrast_volumes.clear()
    
    # Step 3: Process each volume by loading one at a time
    log_progress("Starting registration of volumes to atlas...")
    for key, path in tqdm(volume_paths.items(), desc="Registering volumes"):
        try:
            log_progress(f"Processing volume {key} (phase: {phase_map.get(key, 'unknown')})...")
            moving_img = sitk.ReadImage(path)
            
            # Process through register_to_atlas for all volumes, including atlas
            transform, registered_img = register_to_atlas(atlas_img, moving_img, is_atlas=(key == atlas_key))
            
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
    INPUT_DIR = "utils/debug/ncct_cect/vindr_ds/cropped_volumes"
    OUTPUT_DIR = "utils/debug/ncct_cect/vindr_ds/registered_volumes"
    LABELS_PATH = "utils/debug/ncct_cect/vindr_ds/labels.csv"
    main(INPUT_DIR, OUTPUT_DIR, LABELS_PATH)