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

from volume_utils import resample_to_k_slices, get_crop_bounds, save_image, save_original_volume
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
    """Validate that resampling didn't create significant artifacts.
    Note: This is a basic check and might need refinement.
    """
    try:
        # Convert to numpy arrays, handling potential data type differences
        orig_array = sitk.GetArrayFromImage(original_img)
        resampled_array = sitk.GetArrayFromImage(resampled_img)

        # Calculate mean intensity difference in the overlapping region or a representative area
        # For simplicity, let's check the mean of the whole image for now.
        mean_orig = np.mean(orig_array)
        mean_resampled = np.mean(resampled_array)
        
        if mean_orig == 0:
            # Avoid division by zero if the original image is all zeros
            if mean_resampled != 0:
                 logging.warning("Original image mean is zero, but resampled is not.")
                 return False
            return True # Both are zero

        mean_diff = np.abs(mean_orig - mean_resampled)
        mean_diff_percent = (mean_diff / mean_orig) * 100
        
        logging.info(f"Resampling validation - Mean diff: {mean_diff:.2f}, Mean diff percent: {mean_diff_percent:.2f}%")

        if mean_diff_percent > max_diff_percent:
            logging.warning(f"Resampling created significant intensity changes: {mean_diff_percent:.2f}% > {max_diff_percent}%")
            return False
        return True
    except Exception as e:
        logging.error(f"Error during resampling validation: {e}")
        return False # Assume validation failed on error

def validate_dimensions(img, expected_size):
    """Validate that the image has the expected dimensions (depth, height, width)."""
    actual_size = img.GetSize() # SimpleITK size is (width, height, depth)
    # We need to compare with expected_size (depth, height, width)
    # Let's assume expected_size is (depth, height, width) as used in your code comments
    # and SimpleITK size is (x, y, z) = (width, height, depth)
    # So, SimpleITK size (2), (1), (0) should match expected_size (0), (1), (2) respectively.

    if len(actual_size) != 3 or len(expected_size) != 3:
         logging.error(f"Dimension validation failed: Invalid size lengths. Actual: {len(actual_size)}, Expected: {len(expected_size)}")
         return False

    # Compare depths (SimpleITK size[2] with expected_size[0])
    if actual_size[2] != expected_size[0]:
        logging.error(f"Dimension validation failed: Depth mismatch. Expected: {expected_size[0]}, Got: {actual_size[2]}")
        return False
    
    # Compare heights (SimpleITK size[1] with expected_size[1])
    if actual_size[1] != expected_size[1]:
        logging.error(f"Dimension validation failed: Height mismatch. Expected: {expected_size[1]}, Got: {actual_size[1]}")
        return False

    # Compare widths (SimpleITK size[0] with expected_size[2])
    if actual_size[0] != expected_size[2]:
         logging.error(f"Dimension validation failed: Width mismatch. Expected: {expected_size[2]}, Got: {actual_size[0]}")
         return False

    logging.info(f"Dimension validation successful. Actual size (width, height, depth): {actual_size}, Expected size (depth, height, width): ({expected_size[0]}, {expected_size[1]}, {expected_size[2]}) ")
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

# Load DICOM series with metadata using the updated function
print("Starting to load DICOM series with metadata...")
dicom_series_data = load_and_cache_dicom_series(BATCH_DIR, LABELS_CSV, CACHE_PATH)
print(f"DICOM series loaded successfully. Number of series: {len(dicom_series_data)}")

# Step 1: Resample all volumes and save them, preserving metadata
print("\nStep 1: Starting resampling of all volumes with metadata preservation")
resampled_paths = []

# Iterate over the loaded data, which now includes metadata
for i, (study_uid, series_uid, volume, metadata) in enumerate(tqdm(dicom_series_data)):
    print(f"\nProcessing volume {i+1}/{len(dicom_series_data)}: {study_uid}_{series_uid}")
    
    # Save original volume with metadata
    print(f"Saving original volume...")
    filename = f"{study_uid}_{series_uid}_original"
    save_original_volume(volume, ORIGINAL_DIR, filename)
    # Convert to numpy for abdomen cropping
    # Make sure to use the SimpleITK image volume here
    volume_array = sitk.GetArrayFromImage(volume)
    
    # Crop abdomen region
    print("Cropping abdomen region...")
    try:
        # crop_abdomen_region expects numpy array and now returns array and bounds
        cropped_array, (min_coords, max_coords) = crop_abdomen_region(volume_array, lower_hu=-200, upper_hu=300, margin=20)
        
        # Convert cropped array back to SimpleITK image
        cropped_volume = sitk.GetImageFromArray(cropped_array)
        
        # Calculate the new origin based on the original origin, spacing, and cropping bounds
        original_origin = volume.GetOrigin()
        original_spacing = volume.GetSpacing()
        
        # The new origin is the original origin plus the displacement of the minimum cropped corner
        # The indices min_coords are in (Z, Y, X) order corresponding to SimpleITK's numpy array order
        # SimpleITK Origin and Spacing are in (X, Y, Z) order
        
        # Displacement in X, Y, Z
        # Note: min_coords are 0-indexed numpy array indices
        # SimpleITK origin and spacing are 0-indexed as well in their respective dimensions
        displacement_x = min_coords[2] * original_spacing[0]
        displacement_y = min_coords[1] * original_spacing[1]
        displacement_z = min_coords[0] * original_spacing[2]
        
        new_origin = (
            original_origin[0] + displacement_x,
            original_origin[1] + displacement_y,
            original_origin[2] + displacement_z
        )

        # Set spatial information manually on the cropped volume
        cropped_volume.SetSpacing(volume.GetSpacing())
        cropped_volume.SetDirection(volume.GetDirection())
        cropped_volume.SetOrigin(new_origin)
        
        print(f"Successfully cropped abdomen region. New origin: {new_origin}")
    except ValueError as e:
        print(f"Warning: {e}. Skipping abdomen cropping and using original volume.")
        # If cropping fails (e.g., no abdomen region found), use the original volume
        cropped_volume = volume
        min_coords = (0, 0, 0) # Set bounds to (0,0,0) if cropping was skipped
        max_coords = volume_array.shape # Set bounds to original shape if cropping was skipped

    
    # Resample the cropped volume (or original volume if cropping failed) to target slices
    print(f"Resampling volume to {TARGET_SLICE_NUM} slices using B-spline interpolation...")
    
    # The resample_to_k_slices function in volume_utils.py is updated to use B-spline for volumes.
    # No need to set interpolator explicitly here unless overriding.
    vol_resampled = resample_to_k_slices(cropped_volume, TARGET_SLICE_NUM)
    
    # Save resampled volume with metadata
    print(f"Saving resampled volume...")
    resampled_save_name = f"{study_uid}_{series_uid}_resampled"
    resampled_path = os.path.join(RESAMPLED_DIR, f"{resampled_save_name}.nii.gz")
    save_image(vol_resampled, RESAMPLED_DIR, resampled_save_name, "volume", metadata=metadata)
    resampled_paths.append(resampled_path)
    print(f"Completed processing volume {i+1}/{len(dicom_series_data)}")

print(f"\nStep 1 completed. Processed {len(resampled_paths)} volumes")

# Step 2: Update dataset_0.json with all paths
print("Step 2: Updating dataset_0.json")
dataset_config = {
    # Store only the filename in the JSON
    "testing": [{"image": os.path.basename(path)} for path in resampled_paths]
}
dataset_json_path = os.path.join("bundles/multi_organ_segmentation/configs/dataset_0.json")
with open(dataset_json_path, 'w') as f:
    json.dump(dataset_config, f, indent=4) # Use indent for readability
print(f"Updated dataset_0.json with {len(resampled_paths)} paths")

# Step 3: Run segmentation inference
logging.info("Step 3: Running segmentation inference")
logging.info(f"Running segmentation on device: {device}")
config_file = "bundles/multi_organ_segmentation/configs/inference.yaml"
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

# Step 4: Process segmentation masks and crop volumes
logging.info("Step 4: Processing segmentation masks and cropping volumes")
slice_counts = []
valid_volumes = []

# Iterate through the original DICOM series data to match with segmentation masks
# We need study_uid and series_uid to find the corresponding segmentation masks

for i, (study_uid, series_uid, original_volume, original_metadata) in enumerate(tqdm(dicom_series_data)):
    print(f"\nProcessing segmentation and cropping for volume {i+1}/{len(dicom_series_data)}: {study_uid}_{series_uid}")

    # Load the resampled volume (saved in Step 1)
    resampled_save_name = f"{study_uid}_{series_uid}_resampled"
    resampled_path = os.path.join(RESAMPLED_DIR, f"{resampled_save_name}.nii.gz")
    
    if not os.path.exists(resampled_path):
        logging.error(f"Resampled volume not found at {resampled_path}. Skipping cropping for this volume.")
        continue

    try:
        vol_resampled = sitk.ReadImage(resampled_path)
    except Exception as e:
        logging.error(f"Error reading resampled volume {resampled_path}: {e}. Skipping.")
        continue

    # Get corresponding segmentation mask path generated in Step 3
    # Assuming the segmentation output is in SEGMENTATION_DIR with a consistent naming convention
    seg_name = f"{resampled_save_name}_pred.nii.gz" # Assuming MONAI bundle output naming
    seg_path = os.path.join(SEGMENTATION_DIR, seg_name)
    
    if not os.path.exists(seg_path):
        # Try alternative naming convention if needed
        seg_name_alt = f"{study_uid}_{series_uid}_resampled_pred.nii.gz"
        seg_path_alt = os.path.join("runs/segmentation_output", seg_name_alt) # Assuming a common MONAI output dir
        
        if os.path.exists(seg_path_alt):
            seg_path = seg_path_alt
            logging.info(f"Found segmentation mask at alternative path: {seg_path_alt}")
        else:
            logging.error(f"Segmentation mask not found for {study_uid}_{series_uid} at {seg_path} or {seg_path_alt}. Skipping.")
            continue
        
    try:
        # Load segmentation mask
        seg_mask_sitk = sitk.ReadImage(seg_path)
        # Ensure segmentation mask has the same spatial information as the resampled volume
        # This is crucial for correct alignment during cropping.
        try:
             seg_mask_sitk.CopyInformation(vol_resampled)
        except Exception as e:
             logging.warning(f"Could not copy spatial information to segmentation mask: {e}")
             # Attempt to set information manually if CopyInformation fails
             try:
                 seg_mask_sitk.SetSpacing(vol_resampled.GetSpacing())
                 seg_mask_sitk.SetOrigin(vol_resampled.GetOrigin())
                 seg_mask_sitk.SetDirection(vol_resampled.GetDirection())
             except Exception as e_manual:
                  logging.warning(f"Could not manually set spatial information for segmentation mask: {e_manual}")

        seg_mask = sitk.GetArrayFromImage(seg_mask_sitk) # Convert to numpy array (Z, H, W)

    except Exception as e:
        logging.error(f"Error reading segmentation mask {seg_path}: {e}. Skipping.")
        continue

    # Get crop bounds from the segmentation mask
    # get_crop_bounds expects numpy array with Z, H, W
    zmin, zmax = get_crop_bounds(seg_mask, INCLUDED_LABELS, z_margin=Z_MARGIN)
    logging.info(f"Crop bounds for {study_uid}_{series_uid} - z_min: {zmin}, z_max: {zmax}, crop size: {zmax - zmin}")
    
    # Validate crop size
    if not validate_crop_size(zmin, zmax):
        print(f"Crop size validation failed for {study_uid}_{series_uid}. Skipping.")
        continue
    
    if zmax <= zmin:
        print(f"Invalid crop bounds for {study_uid}_{series_uid}: z_min={zmin}, z_max={zmax}. Skipping.")
        continue
    
    # Crop both segmentation mask and volume using the calculated bounds
    # Numpy slicing: array[z_slice_min:z_slice_max, :, :]
    cropped_seg_array = seg_mask[zmin:zmax]
    cropped_vol_array = sitk.GetArrayFromImage(vol_resampled)[zmin:zmax]
    
    # Convert cropped numpy arrays back to SimpleITK images
    # Crucially, copy the spatial information from the original resampled volume
    cropped_seg_sitk = sitk.GetImageFromArray(cropped_seg_array.astype(np.uint8))
    cropped_vol_sitk = sitk.GetImageFromArray(cropped_vol_array.astype(np.int16)) # Maintain int16 for volume
    
    # Copy information from the resampled volume BEFORE cropping, as cropping changes the origin/size
    # A better approach might be to calculate the new origin after cropping and set it.
    # For now, let's copy information and then adjust origin based on crop.
    
    # Calculate the new origin after cropping
    original_origin = vol_resampled.GetOrigin()
    original_spacing = vol_resampled.GetSpacing()
    # New origin is the original origin plus the displacement in the z-direction due to cropping
    new_origin_z = original_origin[2] + zmin * original_spacing[2]
    new_origin = (original_origin[0], original_origin[1], new_origin_z)

    cropped_seg_sitk.SetSpacing(vol_resampled.GetSpacing())
    cropped_seg_sitk.SetDirection(vol_resampled.GetDirection())
    cropped_seg_sitk.SetOrigin(new_origin)

    cropped_vol_sitk.SetSpacing(vol_resampled.GetSpacing())
    cropped_vol_sitk.SetDirection(vol_resampled.GetDirection())
    cropped_vol_sitk.SetOrigin(new_origin)

    # Save cropped segmentation with metadata (optional for segmentation, but good practice)
    cropped_seg_save_name = f"{study_uid}_{series_uid}_cropped_segmentation"
    # Passing original metadata might be confusing for a cropped segmentation.
    # Let's pass a simplified metadata dictionary or None.
    # For now, pass the original metadata.
    save_image(cropped_seg_sitk, CROPPED_DIR, cropped_seg_save_name, "segmentation", metadata=original_metadata)
    
    # Save cropped volume with metadata
    cropped_vol_save_name = f"{study_uid}_{series_uid}_cropped_volume"
    save_image(cropped_vol_sitk, CROPPED_DIR, cropped_vol_save_name, "volume", metadata=original_metadata)
    
    slice_counts.append(zmax - zmin)
    valid_volumes.append((study_uid, series_uid, zmax - zmin))

logging.info(f"Step 4 completed. Processed {len(valid_volumes)} volumes for cropping.")

# Step 5: Compute average cropped length and final resample
if slice_counts:
    avg_slices = int(np.round(np.mean(slice_counts)))
    print(f"Average cropped length: {avg_slices} slices")
    
    # Define expected final size after resampling to average slices
    # Get size from one of the cropped volumes before final resampling
    if valid_volumes:
        # Get the first successfully cropped volume's path
        first_cropped_vol_name = f"{valid_volumes[0][0]}_{valid_volumes[0][1]}_cropped_volume"
        first_cropped_vol_path = os.path.join(CROPPED_DIR, f"{first_cropped_vol_name}.nii.gz")
        try:
            first_cropped_img = sitk.ReadImage(first_cropped_vol_path)
            # SimpleITK size is (width, height, depth)
            expected_final_size_sitk = (first_cropped_img.GetSize()[0], first_cropped_img.GetSize()[1], avg_slices)
            # Our validation function expects (depth, height, width)
            expected_final_size_validation = (avg_slices, first_cropped_img.GetSize()[1], first_cropped_img.GetSize()[0])
            print(f"Expected final SimpleITK size (width, height, depth): {expected_final_size_sitk}")
            print(f"Expected final validation size (depth, height, width): {expected_final_size_validation}")

        except Exception as e:
            logging.error(f"Could not read first cropped volume for expected size determination: {e}. Skipping final validation.")
            expected_final_size_validation = None # Disable dimension validation
    else:
        print("No valid volumes processed for cropping. Skipping final resampling and validation.")
        expected_final_size_validation = None # Disable dimension validation


    # Re-resample all saved cropped volumes to avg_slices
    print("Step 5: Final resampling of cropped volumes to average length")
    final_processed_volumes_count = 0
    
    # Iterate through files in the CROPPED_DIR that are volumes
    cropped_volume_files = [f for f in os.listdir(CROPPED_DIR) if f.endswith('_cropped_volume.nii.gz')]

    if not cropped_volume_files:
        print("No cropped volume files found for final resampling. Skipping Step 5.")
    else:
        for fname in tqdm(cropped_volume_files):
            img_path = os.path.join(CROPPED_DIR, fname)
            
            try:
                img = sitk.ReadImage(img_path)
                original_size = img.GetSize() # SimpleITK size (width, height, depth)
                original_metadata = {key: img.GetMetaData(key) for key in img.GetMetaDataKeys()} # Extract existing metadata

                print(f"Resampling {fname} from SimpleITK size {original_size} to {avg_slices} slices depth")
                
                # Store original image for validation (optional)
                original_img_for_validation = img
                
                # Perform resampling to average slices depth
                # resample_to_k_slices expects SimpleITK image and target k slices (depth)
                img_resampled_final = resample_to_k_slices(img, avg_slices)
                
                # Validate resampling (optional)
                # Note: This validation might be sensitive after multiple resampling/cropping steps.
                # It compares mean intensity, which might change slightly.
                # if not validate_resampling(original_img_for_validation, img_resampled_final):
                #     print(f"WARNING - Final resampling validation failed for {fname}. Proceeding but noting potential issue.")
                    
                # Validate dimensions against the expected final size
                # validate_dimensions expects SimpleITK image and expected size (depth, height, width)
                if expected_final_size_validation is not None and not validate_dimensions(img_resampled_final, expected_final_size_validation):
                    print(f"ERROR - Final dimension validation failed for {fname}. Skipping save.")
                    continue # Skip saving if dimension validation fails
                
                # Save the final resampled image, preserving metadata
                sitk.WriteImage(img_resampled_final, img_path, useCompression=True) # Overwrite the cropped volume with the final resampled one
                logging.info(f"Final resampled volume saved to: {img_path} with SimpleITK size {img_resampled_final.GetSize()}")
                final_processed_volumes_count += 1
                
            except Exception as e:
                logging.error(f"Error processing {fname} during final resampling: {e}. Skipping.")
                continue

        # Log summary
        print(f"Step 5 completed. Successfully processed {final_processed_volumes_count} out of {len(cropped_volume_files)} cropped volumes for final resampling.")
        if final_processed_volumes_count < len(cropped_volume_files):
            logging.warning(f"Failed to process {len(cropped_volume_files) - final_processed_volumes_count} cropped volumes during final resampling.")

else:
    print("No valid volumes were successfully cropped. Skipping Step 5 (final resampling).")

print("Preprocessing completed.")
