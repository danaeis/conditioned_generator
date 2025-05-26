import os
import json
import pickle
import numpy as np
import torch
import SimpleITK as sitk
from tqdm import tqdm
import logging
import subprocess
import yaml
import sys

# Add device logging at the start
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Using device: {device}")
if torch.cuda.is_available():
    logging.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logging.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

from volume_utils import resample_to_k_slices, get_crop_bounds, save_image
from volume_utils import crop_abdomen_region
from monai.bundle import ConfigParser
from monai.transforms import Compose, LoadImage, EnsureChannelFirst, ScaleIntensity
from monai.data import NibabelReader

from monai.inferers import sliding_window_inference
from monai.transforms import Compose, ScaleIntensity, EnsureChannelFirst, Resize
from monai.networks.nets import UNet
from load_and_cache_dicom_series import load_and_cache_dicom_series

# Add validation functions
def validate_crop_size(zmin, zmax, min_slices=20):
    """Validate that the cropped region is not too small."""
    if zmax - zmin < min_slices:
        logging.warning(f"Crop size {zmax - zmin} is smaller than minimum required {min_slices}")
        return False
    return True

def validate_resampling(original_img, resampled_img, max_diff_percent=5.0):
    """Validate that resampling didn't create significant artifacts."""
    # Convert to numpy arrays
    orig_array = sitk.GetArrayFromImage(original_img)
    resampled_array = sitk.GetArrayFromImage(resampled_img)
    
    # Calculate mean intensity difference
    mean_diff = np.abs(np.mean(orig_array) - np.mean(resampled_array))
    mean_diff_percent = (mean_diff / np.mean(orig_array)) * 100
    
    if mean_diff_percent > max_diff_percent:
        logging.warning(f"Resampling created significant intensity changes: {mean_diff_percent:.2f}%")
        return False
    return True

def validate_dimensions(img, expected_size):
    """Validate that the image has the expected dimensions."""
    actual_size = img.GetSize()
    if actual_size != expected_size:
        logging.error(f"Invalid dimensions. Expected {expected_size}, got {actual_size}")
        return False
    return True

# --- CONFIG ---
BATCH_DIR = '../../ncct_cect/vindr_ds/batches'
LABELS_CSV = '../../ncct_cect/vindr_ds/labels.csv'
CACHE_PATH = '../../ncct_cect/vindr_ds/cached_dicom_series.pkl'

ALIGNED_SLICES_DIR = 'debug/ncct_cect/vindr_ds/aligned_slices'
CROPPED_DIR = 'debug/ncct_cect/vindr_ds/cropped_volumes'
RESAMPLED_DIR = 'debug/ncct_cect/vindr_ds/resampled_volumes'
SEGMENTATION_DIR = 'debug/ncct_cect/vindr_ds/segmentation_masks'
ORIGINAL_DIR = 'debug/ncct_cect/vindr_ds/original_volumes'

STANDARD_SPACING = (1.0, 1.0, 1.0)
TARGET_SLICE_NUM = 128

# Labels to keep for cropping
INCLUDED_LABELS = {1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13}
Z_MARGIN = 5

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('preprocessing.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Force immediate flushing of stdout
sys.stdout.reconfigure(line_buffering=True)

# Create all necessary directories
for dir_path in [CROPPED_DIR, RESAMPLED_DIR, SEGMENTATION_DIR, ORIGINAL_DIR]:
    os.makedirs(dir_path, exist_ok=True)
    logging.info(f"Created directory: {dir_path}")

# Load DICOM series
print("Starting to load DICOM series...")
print("Starting to load DICOM series...")
dicom_series, z_spacings = load_and_cache_dicom_series(BATCH_DIR, LABELS_CSV, CACHE_PATH)
print(f"DICOM series loaded successfully. Number of series: {len(dicom_series)}")
print(f"DICOM series loaded successfully. Number of series: {len(dicom_series)}")

# Step 1: Resample all volumes and save them
print("\nStep 1: Starting resampling of all volumes")
print("Step 1: Starting resampling of all volumes")
resampled_paths = []
for i, (study_uid, series_uid, volume) in enumerate(tqdm(dicom_series)):
    print(f"\nProcessing volume {i+1}/{len(dicom_series)}: {study_uid}_{series_uid}")
    
    # Save original volume
    print(f"Saving original volume...")
    original_save_name = f"{study_uid}_{series_uid}_original"
    save_image(volume, ORIGINAL_DIR, original_save_name, "volume")
    
    # Convert to numpy for abdomen cropping
    volume_array = sitk.GetArrayFromImage(volume)
    
    # Crop abdomen region
    print("Cropping abdomen region...")
    try:
        cropped_array = crop_abdomen_region(volume_array, lower_hu=-200, upper_hu=300, margin=20)
        volume = sitk.GetImageFromArray(cropped_array)
        volume.CopyInformation(sitk.ReadImage(os.path.join(ORIGINAL_DIR, f"{original_save_name}.nii.gz")))
        print("Successfully cropped abdomen region")
    except ValueError as e:
        print(f"Warning: {e}. Using original volume.")
    
    # Resample to target slices
    print(f"Resampling volume to {TARGET_SLICE_NUM} slices...")
    vol_resampled = resample_to_k_slices(volume, TARGET_SLICE_NUM)
    
    # Save resampled volume
    print(f"Saving resampled volume...")
    resampled_save_name = f"{study_uid}_{series_uid}_resampled"
    resampled_path = os.path.join(RESAMPLED_DIR, f"{resampled_save_name}.nii.gz")
    save_image(vol_resampled, RESAMPLED_DIR, resampled_save_name, "volume")
    resampled_paths.append(resampled_path)
    print(f"Completed processing volume {i+1}/{len(dicom_series)}")

print(f"\nStep 1 completed. Processed {len(resampled_paths)} volumes")
print(f"Step 1 completed. Processed {len(resampled_paths)} volumes")

# Step 2: Update dataset_0.json with all paths
print("Step 2: Updating dataset_0.json")
dataset_config = {
    "testing": [{"image": path.split("/")[-1]} for path in resampled_paths]
}
dataset_json_path = os.path.join("bundles/multi_organ_segmentation/configs/dataset_0.json")
with open(dataset_json_path, 'w') as f:
    json.dump(dataset_config, f)
print(f"Updated dataset_0.json with {len(resampled_paths)} paths")

# Step 3: Run segmentation inference
print("Step 3: Running segmentation inference")
print(f"Running segmentation on device: {device}")
config_file = "bundles/multi_organ_segmentation/configs/inference.yaml"
cmd = [
    "python", "-m", "monai.bundle", "run",
    "--config_file", config_file
]

try:
    result = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True
    )
    print("Segmentation completed successfully")
    print("Command stdout:")
    print(result.stdout)
    if result.stderr:
        logging.warning("Command stderr:")
        logging.warning(result.stderr)
except subprocess.CalledProcessError as e:
    print(f"Error running segmentation: {e}")
    if e.stdout:
        print(f"Command stdout: {e.stdout}")
    if e.stderr:
        print(f"Command stderr: {e.stderr}")
    raise

# Step 4: Process segmentation masks and crop volumes
logging.info("Step 4: Processing segmentation masks and cropping volumes")
slice_counts = []
valid_volumes = []

for resampled_path in tqdm(resampled_paths):
    study_uid = os.path.basename(resampled_path).split('_')[0]
    series_uid = os.path.basename(resampled_path).split('_')[1]
    
    # Load resampled volume
    vol_resampled = sitk.ReadImage(resampled_path)
    
    # Get corresponding segmentation mask
    seg_name = f"{os.path.basename(resampled_path).replace('.nii.gz', '')}_pred.nii.gz"
    seg_path = os.path.join(SEGMENTATION_DIR, seg_name)
    
    if not os.path.exists(seg_path):
        logging.error(f"Segmentation mask not found at {seg_path}")
        continue
        
    # Load segmentation mask
    seg_mask = sitk.GetArrayFromImage(sitk.ReadImage(seg_path))
    
    # Get crop bounds
    zmin, zmax = get_crop_bounds(seg_mask, INCLUDED_LABELS, z_margin=Z_MARGIN)
    logging.info(f"Crop bounds for {study_uid}_{series_uid} - z_min: {zmin}, z_max: {zmax}")
    
    # Validate crop size
    if not validate_crop_size(zmin, zmax):
        print(f"Crop size validation failed for {study_uid}_{series_uid}")
        continue
    
    if zmax <= zmin:
        print(f"Invalid crop bounds for {study_uid}_{series_uid}: z_min={zmin}, z_max={zmax}")
        continue
    
    # Crop both segmentation mask and volume
    cropped_seg = seg_mask[zmin:zmax]
    cropped_vol = sitk.GetArrayFromImage(vol_resampled)[zmin:zmax]
    
    # Save cropped segmentation
    cropped_seg_save_name = f"{study_uid}_{series_uid}_cropped_segmentation"
    save_image(cropped_seg, CROPPED_DIR, cropped_seg_save_name, "segmentation", vol_resampled)
    
    # Save cropped volume
    cropped_vol_save_name = f"{study_uid}_{series_uid}_cropped_volume"
    save_image(cropped_vol, CROPPED_DIR, cropped_vol_save_name, "volume", vol_resampled)
    
    slice_counts.append(zmax - zmin)
    valid_volumes.append((study_uid, series_uid, zmax - zmin))

# Step 5: Compute average cropped length and resample
if slice_counts:
    avg_slices = int(np.round(np.mean(slice_counts)))
    print(f"Average cropped length: {avg_slices} slices")
    
    # Define expected final size
    first_img = sitk.ReadImage(os.path.join(CROPPED_DIR, os.listdir(CROPPED_DIR)[0]))
    expected_size = (avg_slices, first_img.GetSize()[1], first_img.GetSize()[2])
    print(f"Expected final size for all volumes: {expected_size}")

    # Re-resample all saved volumes to avg_slices
    print("Step 5: Final resampling to average length")
    final_valid_volumes = []
    
    for fname in os.listdir(CROPPED_DIR):
        img_path = os.path.join(CROPPED_DIR, fname)
        img = sitk.ReadImage(img_path)
        original_size = img.GetSize()
        print(f"Resampling {fname} from size {original_size} to {avg_slices} slices")
        
        # Store original image for validation
        original_img = img
        
        # Perform resampling
        img_resampled = resample_to_k_slices(img, avg_slices)
        
        # Validate resampling
        if not validate_resampling(original_img, img_resampled):
            print(f"ERROR - Resampling validation failed for {fname}")
            continue
            
        # Validate dimensions
        if not validate_dimensions(img_resampled, expected_size):
            print(f"ERROR - Dimension validation failed for {fname}")
            continue
        
        # Save validated resampled image
        sitk.WriteImage(img_resampled, img_path)
        print(f"Final size after resampling: {img_resampled.GetSize()}")
        final_valid_volumes.append(fname)
    
    # Log summary
    print(f"Successfully processed {len(final_valid_volumes)} out of {len(valid_volumes)} volumes")
    if len(final_valid_volumes) < len(valid_volumes):
        logging.warning(f"Failed to process {len(valid_volumes) - len(final_valid_volumes)} volumes")
else:
    print("No valid segmentations were produced. Cannot compute average slice count.")

print("Preprocessing completed successfully")
