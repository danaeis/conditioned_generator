import os
import json
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm
from pathlib import Path
import pandas as pd
from registration_utils import (
    register_to_atlas, compute_quick_metric, parallel_compute_metric
)
import logging
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import time

# Get the workspace root directory (parent of utils)
WORKSPACE_ROOT = str(Path(__file__).parent)

# Log SimpleITK version
logging.info(f"SimpleITK Version: {sitk.Version().VersionString()}")

# Set up logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(WORKSPACE_ROOT, 'registration.log'))
    ]
)

# Log system information
logging.info(f"Number of CPU cores: {multiprocessing.cpu_count()}")
logging.info(f"Python process ID: {os.getpid()}")

# --- CONFIG ---
INPUT_VOLUMES_DIR = os.path.join(WORKSPACE_ROOT, 'debug/ncct_cect/vindr_ds/original_volumes')
REGISTERED_DIR = os.path.join(WORKSPACE_ROOT, 'debug/ncct_cect/vindr_ds/registered_volumes')
ATLAS_SAVE_PATH = os.path.join(WORKSPACE_ROOT, 'debug/ncct_cect/vindr_ds/atlas/reference.nii.gz')
LABELS_CSV = os.path.join(WORKSPACE_ROOT, 'debug/ncct_cect/vindr_ds/labels.csv')

# Number of worker threads for parallel processing
NUM_WORKERS = 2  # Reduced number of workers to avoid memory issues

# Create necessary directories
os.makedirs(REGISTERED_DIR, exist_ok=True)
os.makedirs(os.path.dirname(ATLAS_SAVE_PATH), exist_ok=True)

# Log the paths being used
logging.info(f"Workspace root: {WORKSPACE_ROOT}")
logging.info(f"Input volumes directory: {INPUT_VOLUMES_DIR}")
logging.info(f"Output directory: {REGISTERED_DIR}")
logging.info(f"Atlas save path: {ATLAS_SAVE_PATH}")
logging.info(f"Labels CSV: {LABELS_CSV}")

# --- STEP 1: Load label info ---
labels_df = pd.read_csv(LABELS_CSV)
labels_df = labels_df.dropna(subset=["StudyInstanceUID", "SeriesInstanceUID", "Label"])
phase_labels = {
    (row["StudyInstanceUID"], row["SeriesInstanceUID"]): row["Label"].lower()
    for _, row in labels_df.iterrows()
}

# --- STEP 2: Find Non-Contrast Volumes ---
print("Finding non-contrast volumes...")
volume_paths = sorted(list(Path(INPUT_VOLUMES_DIR).rglob("*.nii.gz")))
logging.info(f"Found {len(volume_paths)} total volume files")

non_contrast_paths = []
for p in volume_paths:
    name = p.stem
    try:
        study_uid, series_uid = name.split("_")[:2]
    except ValueError:
        logging.warning(f"Skipping malformed filename: {name}")
        continue
    
    phase = phase_labels.get((study_uid, series_uid), None)
    if phase is None:
        logging.warning(f"Skipping {name} (no label info)")
        continue
    
    if "non" in phase:
        non_contrast_paths.append((p, study_uid, series_uid))
        logging.info(f"Found non-contrast volume: {name}")

if not non_contrast_paths:
    raise ValueError("No non-contrast cases found for atlas selection.")

logging.info(f"Found {len(non_contrast_paths)} non-contrast volumes as atlas candidates")

# --- STEP 3: Select Best Atlas from Non-Contrast ---
print("Selecting atlas from non-contrast images using parallel processing...")

def compute_pairwise_metrics():
    """Compute pairwise metrics between all non-contrast images in parallel."""
    metric_tasks = []
    for i, (fixed_path, _, _) in enumerate(non_contrast_paths):
        for j, (moving_path, _, _) in enumerate(non_contrast_paths):
            if i != j:
                metric_tasks.append((fixed_path, moving_path, 'ncc'))
    
    all_metrics = []
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(parallel_compute_metric, task) for task in metric_tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Computing metrics"):
            all_metrics.append(future.result())
    
    return all_metrics

# Compute metrics in parallel
metrics = compute_pairwise_metrics()
metrics = np.array(metrics).reshape(len(non_contrast_paths), -1)
mean_metrics = np.mean(metrics, axis=1)
best_idx = np.argmin(mean_metrics)

# Load and save the best atlas
atlas_path = non_contrast_paths[best_idx][0]
atlas_image = sitk.ReadImage(str(atlas_path))
logging.info(f"Selected atlas: {atlas_path.name} with mean metric: {mean_metrics[best_idx]:.4f}")
sitk.WriteImage(atlas_image, ATLAS_SAVE_PATH)

# --- STEP 4: Register All Volumes to Atlas ---
print(f"\nRegistering all volumes to selected atlas...")

def register_volume(args):
    """Helper function for parallel volume registration."""
    p, study_uid, series_uid = args
    start_time = time.time()
    try:
        logging.info(f"Starting registration for {p.name}")
        logging.info(f"Reading image from {p}")
        moving_image = sitk.ReadImage(str(p))
        logging.info(f"Successfully read image {p.name}")
        
        # Log image sizes
        logging.info(f"Atlas size: {atlas_image.GetSize()}, Moving image size: {moving_image.GetSize()}")
        
        logging.info(f"Starting registration process for {p.name}")
        transform, registered_image = register_to_atlas(
            fixed_image=atlas_image,
            moving_image=moving_image,
            transform_type='multi_step'
        )
        logging.info(f"Registration completed for {p.name}")
        
        # Save registered image
        filename = f"{study_uid}_{series_uid}_registered.nii.gz"
        save_path = os.path.join(REGISTERED_DIR, filename)
        logging.info(f"Saving registered image to {save_path}")
        sitk.WriteImage(registered_image, save_path)
        
        elapsed_time = time.time() - start_time
        logging.info(f"Successfully registered {p.name} in {elapsed_time:.2f} seconds")
        return True
    except Exception as e:
        elapsed_time = time.time() - start_time
        logging.error(f"Error registering {p.name} after {elapsed_time:.2f} seconds: {str(e)}")
        logging.error(f"Full error details:", exc_info=True)
        return False
    finally:
        if 'moving_image' in locals():
            del moving_image
        if 'registered_image' in locals():
            del registered_image
        gc.collect()

# Prepare registration tasks
registration_tasks = []
for p in volume_paths:
    name = p.stem
    try:
        study_uid, series_uid = name.split("_")[:2]
    except ValueError:
        continue
    
    phase = phase_labels.get((study_uid, series_uid), None)
    if phase is None:
        continue
    
    registration_tasks.append((p, study_uid, series_uid))

logging.info(f"Starting registration of {len(registration_tasks)} volumes with {NUM_WORKERS} workers")

try:
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = []
        for task in registration_tasks:
            future = executor.submit(register_volume, task)
            futures.append(future)
            logging.info(f"Submitted registration task for {task[0].name}")
        
        results = []
        for future in tqdm(as_completed(futures), total=len(futures), desc="Registering volumes"):
            try:
                result = future.result(timeout=3600)  # 1 hour timeout per volume
                results.append(result)
                # Log progress
                success_count = sum(1 for r in results if r)
                logging.info(f"Progress: {len(results)}/{len(registration_tasks)} volumes processed, {success_count} successful")
            except Exception as e:
                logging.error(f"Error in registration task: {str(e)}")
                results.append(False)
except Exception as e:
    logging.error(f"Error in parallel processing: {str(e)}")
    raise

success_count = sum(1 for r in results if r)
logging.info(f"Registration complete. Successfully registered {success_count} out of {len(registration_tasks)} volumes")

print("Registration complete. All images saved to:", REGISTERED_DIR)
