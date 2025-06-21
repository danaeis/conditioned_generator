import os
import json
import logging
import subprocess
import yaml
import sys
import glob
import nibabel as nib
from datetime import datetime
from pathlib import Path

import torch
import pickle
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

from dicom_utils import process_original_volumes
from volume_utils import process_ct_and_crop_abdomen

# def get_crop_bounds(seg_mask, included_labels, z_margin=5):
#     # Create a boolean array where each slice is True if it contains any of the included labels
#     z_slices = np.any(np.isin(seg_mask, list(included_labels)), axis=-1)
#     indices = np.where(z_slices)[0]
#     if len(indices) == 0:
#         # If no slice contains any of the included labels, keep the whole volume
#         logging.warning("No slices found containing included labels, keeping whole volume")
#         return 0, seg_mask.shape[0]
#     # Determine the topmost and bottommost slice that contains at least one of the included labels
#     z_min = max(indices[0] - z_margin, 0)
#     z_max = min(indices[-1] + z_margin + 1, seg_mask.shape[0])  # +1 to include the last slice
#     logging.info(f"Crop bounds - z_min: {z_min}, z_max: {z_max}, crop size: {z_max - z_min}")
#     return z_min, z_max

# Configure logging with absolute path
workspace_root = Path("/media/disk1/saeedeh_danaei/conditioned_generator")
log_dir = workspace_root / "logs"
log_dir.mkdir(exist_ok=True)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = log_dir / f'pre_segCrop_{timestamp}.log'

# Configure logging format and handlers
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(str(log_file)),
        logging.StreamHandler(sys.stdout)
    ]
)

logging.info(f"Log file created at: {log_file}")

# Set PyTorch memory allocation settings
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Add device logging at the start
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Using device: {device}")
if torch.cuda.is_available():
    logging.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logging.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# --- CONFIG ---
BATCH_DIR = '../ncct_cect/vindr_ds/batches'
LABELS_CSV = '../ncct_cect/vindr_ds/labels.csv'
CACHE_PATH = '../ncct_cect/vindr_ds/cached_dicom_series.pkl'

DEBUG_DIR = 'utils/debug/ncct_cect/vindr_ds/'
ALIGNED_SLICES_DIR = DEBUG_DIR + 'aligned_slices'
CROPPED_DIR = DEBUG_DIR + 'cropped_volumes'
RESAMPLED_DIR = DEBUG_DIR + 'resampled_volumes'
SEGMENTATION_DIR = DEBUG_DIR + 'segmentation_masks'
ORIGINAL_DIR = DEBUG_DIR + 'original_volumes'


STANDARD_SPACING = (1.0, 1.0, 1.0)
TARGET_SLICE_NUM = 128

# Labels to keep for cropping
INCLUDED_LABELS = {3. , 4. , 5. , 6. , 7. }
Z_MARGIN = 5

# Step 1: Process original volumes
original_volume_paths = process_original_volumes(
        batch_dir=BATCH_DIR,
        labels_csv=LABELS_CSV,
        nifti_root_dir=ORIGINAL_DIR,
        pkl_path=CACHE_PATH,
        overwrite_nifti=False  # Set to True to force reprocessing
    )

MONAI_DATA_DIR = ORIGINAL_DIR
# Recursively gather all NIfTI files (one per series)
nifti_paths = []
for study_uid in os.listdir(MONAI_DATA_DIR):
    study_path = os.path.join(MONAI_DATA_DIR, study_uid)
    if not os.path.isdir(study_path):
        continue
    for series_file in os.listdir(study_path):
        if series_file.endswith(".nii.gz"):
            full_path = os.path.join(study_path, series_file)
            nifti_paths.append(full_path.split("/")[-2]+"/"+full_path.split("/")[-1])

# Create a mapping of series_id to original volume path for quick lookup
series_to_original_path = {}
for path in nifti_paths:
    series_id = path.split("/")[-1].replace(".nii.gz", "")
    series_to_original_path[series_id] = os.path.join(ORIGINAL_DIR, path)

# Create JSON entries with relative paths or full paths
dataset_config = {
    "testing": [{"image": path} for path in nifti_paths]
}

# Save to dataset_0.json
dataset_json_path = os.path.join("utils/bundles/multi_organ_segmentation/configs/dataset_0.json")
os.makedirs(os.path.dirname(dataset_json_path), exist_ok=True)
with open(dataset_json_path, 'w') as f:
    json.dump(dataset_config, f, indent=4)

print(f"✓ Updated dataset_0.json with {len(nifti_paths)} image paths.")

# Step 3: Run segmentation inference
logging.info("Step 3: Running segmentation inference")
logging.info(f"Running segmentation on device: {device}")
# NOTE: The model weights in model.pt are a raw state_dict, not wrapped in a dict with a 'model' key.
# The inference.yaml config is set to load the weights directly.
config_file = "utils/bundles/multi_organ_segmentation/configs/inference.yaml"
cmd = [
    "python", "-m", "monai.bundle", "run",
    "--config_file", config_file
]

# try:
#     logging.info(f"Running command: {' '.join(cmd)}")
#     result = subprocess.run(
#         cmd,
#         check=True,
#         capture_output=True,
#         text=True
#     )
#     logging.info("Segmentation completed successfully")
#     logging.info("Command stdout:")
#     print(result.stdout) # Print stdout directly to see MONAI progress/messages
#     if result.stderr:
#         logging.warning("Command stderr:")
#         logging.warning(result.stderr) # Log stderr as warning
# except subprocess.CalledProcessError as e:
#     logging.error(f"Error running segmentation: {e}")
#     if e.stdout:
#         logging.error(f"Command stdout: {e.stdout}")
#     if e.stderr:
#         logging.error(f"Command stderr: {e.stderr}")
#     # Re-raise the exception to stop the script if segmentation fails
#     raise
# except FileNotFoundError:
#     logging.error(f"Error: python or monai.bundle command not found. Make sure your environment is set up correctly.")
#     raise
# except Exception as e:
#     logging.error(f"An unexpected error occurred during segmentation inference: {e}")
#     raise

# Step 4: Run cropping
logging.info("Step 4: Running cropping")
logging.info(f"Running cropping on device: {device}")

# Get all segmentation files
seg_files = glob.glob(os.path.join(SEGMENTATION_DIR, "*_seg.nii.gz"))
logging.info(f"Found {len(seg_files)} segmentation files")

for seg_file in seg_files:
    try:
        # Get the original series ID from the segmentation filename
        series_id = os.path.basename(seg_file).replace("_seg.nii.gz", "")
        print(series_id)
        
        # Load segmentation mask
        seg_img = nib.load(seg_file)
        seg_data = seg_img.get_fdata()
        print("unique data in seg data",np.unique(seg_data))
        print("shape", seg_data.shape)
        
        # Find the highest slice that contains any of our included labels
        max_slice = 0
        print("INCLUDED_LABELS:", INCLUDED_LABELS)
        print("Unique values in seg_data:", np.unique(seg_data))

        start_slice = None
        for z in reversed(range(seg_data.shape[2])):  # assuming z-axis is the third dimension
            slice_data = seg_data[:, :, z]
            if np.any(np.isin(slice_data, list(INCLUDED_LABELS))):
                start_slice = z
                break
        
        # Use the pre-computed mapping to get the original volume path
        if series_id in series_to_original_path:
            orig_file = series_to_original_path[series_id]
            if os.path.exists(orig_file):
                orig_img = nib.load(orig_file)
                orig_data = orig_img.get_fdata()
                cropped_orig = orig_data[:, :, :start_slice]
                # find biunding box for abdomen region (remove background)
                cropped_np, cropped_nifti = process_ct_and_crop_abdomen(cropped_orig, orig_img.affine)
                output_orig_path = os.path.join(CROPPED_DIR, f"{series_id}.nii.gz")
                nib.save(cropped_nifti, output_orig_path)
                logging.info(f"Found and cropped original volume for {series_id}")
            else:
                logging.warning(f"Original volume file not found for {series_id}")
        else:
            logging.warning(f"Could not find original volume path for {series_id}")
            
        logging.info(f"Successfully processed {series_id}")
        
    except Exception as e:
        logging.error(f"Error processing {seg_file}: {str(e)}")
        continue

logging.info("Cropping completed")
