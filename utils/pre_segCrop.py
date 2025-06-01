import os
import json
import logging
import subprocess
import yaml
import sys

import torch
import pickle
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

from dicom_utils import process_original_volumes

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
INCLUDED_LABELS = {1, 2, 3, 5, 6, 7}
Z_MARGIN = 5

process_original_volumes(
        batch_dir=BATCH_DIR,
        labels_csv=LABELS_CSV,
        nifti_root_dir=ORIGINAL_DIR,
        pkl_path=CACHE_PATH,
        overwrite_nifti=True  # Set to True to force reprocessing
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

try:
    logging.info(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True
    )
    logging.info("Segmentation completed successfully")
    logging.info("Command stdout:")
    print(result.stdout) # Print stdout directly to see MONAI progress/messages
    if result.stderr:
        logging.warning("Command stderr:")
        logging.warning(result.stderr) # Log stderr as warning
except subprocess.CalledProcessError as e:
    logging.error(f"Error running segmentation: {e}")
    if e.stdout:
        logging.error(f"Command stdout: {e.stdout}")
    if e.stderr:
        logging.error(f"Command stderr: {e.stderr}")
    # Re-raise the exception to stop the script if segmentation fails
    raise
except FileNotFoundError:
    logging.error(f"Error: python or monai.bundle command not found. Make sure your environment is set up correctly.")
    raise
except Exception as e:
    logging.error(f"An unexpected error occurred during segmentation inference: {e}")
    raise
