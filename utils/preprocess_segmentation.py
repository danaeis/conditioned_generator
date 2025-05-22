import os
import json
import pickle
import numpy as np
import torch
import SimpleITK as sitk
from tqdm import tqdm
import logging

from volume_utils import resample_to_k_slices, get_crop_bounds, save_image

from monai.bundle import ConfigParser
from monai.transforms import Compose, LoadImage, EnsureChannelFirst, ScaleIntensity
from monai.data import NibabelReader

from monai.inferers import sliding_window_inference
from monai.transforms import Compose, ScaleIntensity, EnsureChannelFirst, Resize
from monai.networks.nets import UNet  # replace with actual model if different
from load_and_cache_dicom_series import load_and_cache_dicom_series

# --- CONFIG ---
from monai.transforms import (
    LoadImaged,
    EnsureChannelFirst,
    Orientation,
    Spacing,
    ScaleIntensityRange,
    EnsureType,
    Compose,
)

BATCH_DIR = '../../ncct_cect/vindr_ds/batches'
LABELS_CSV = '../../ncct_cect/vindr_ds/labels.csv'
CACHE_PATH = '../../ncct_cect/vindr_ds/cached_dicom_series.pkl'

ALIGNED_SLICES_DIR = 'debug/ncct_cect/vindr_ds/aligned_slices'
CROPPED_DIR = 'debug/ncct_cect/vindr_ds/cropped_volumes'
RESAMPLED_DIR = 'debug/ncct_cect/vindr_ds/resampled_volumes'  # New directory for resampled volumes
SEGMENTATION_DIR = 'debug/ncct_cect/vindr_ds/segmentation_masks'  # New directory for segmentation masks
ORIGINAL_DIR = 'debug/ncct_cect/vindr_ds/original_volumes'  # New directory for original volumes

STANDARD_SPACING = (1.0, 1.0, 1.0)
TARGET_SLICE_NUM = 128

# Labels to keep for cropping (example: BTCV labels excluding arteries)
INCLUDED_LABELS = {1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13}
Z_MARGIN = 5

# Load pretrained MONAI model from bundle path
MODEL_DIR = "bundles/multi_organ_segmentation"
CONFIG_PATH = f"{MODEL_DIR}/configs/inference.yaml"
META_PATH = f"{MODEL_DIR}/configs/metadata.json"

# Placeholder preprocessing transforms
pre_seg_transforms = Compose([
    EnsureChannelFirst(channel_dim=0),  # For arrays
    Orientation(axcodes="RAS"),
    Spacing(pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
    ScaleIntensityRange(a_min=-500, a_max=500, b_min=0.0, b_max=1.0, clip=True),
    EnsureType(),
])

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('preprocessing.log'),
        logging.StreamHandler()
    ]
)

# Create all necessary directories
for dir_path in [CROPPED_DIR, RESAMPLED_DIR, SEGMENTATION_DIR, ORIGINAL_DIR]:
    os.makedirs(dir_path, exist_ok=True)
    logging.info(f"Created directory: {dir_path}")


# Load MONAI model from bundle
config = ConfigParser()
config.read_config(CONFIG_PATH)

# Get the network and preprocessing transforms from config
model = config.get_parsed_content("network")
preprocessing = config.get_parsed_content("preprocessing")

# Move model to GPU and set to eval mode
model.eval().cuda()

dicom_series, z_spacings = load_and_cache_dicom_series(BATCH_DIR, LABELS_CSV, CACHE_PATH)
logging.info("DICOM series loaded successfully")

slice_counts = []

for study_uid, series_uid, volume in tqdm(dicom_series):
    logging.info(f"\nProcessing volume: {study_uid}_{series_uid}")
    
    # Save original volume
    original_save_name = f"{study_uid}_{series_uid}_original"
    save_image(volume, ORIGINAL_DIR, original_save_name, "volume")
    
    # Step 1: Resample to target slices
    logging.info("Step 1: Resampling volume")
    vol_resampled = resample_to_k_slices(volume, TARGET_SLICE_NUM)
    
    # Save resampled volume
    resampled_save_name = f"{study_uid}_{series_uid}_resampled"
    save_image(vol_resampled, RESAMPLED_DIR, resampled_save_name, "volume")
    
    resampled_path = os.path.join(RESAMPLED_DIR, f"{resampled_save_name}.nii.gz")
    # Load the image first
    vol_sitk = sitk.ReadImage(resampled_path)
    vol_np = sitk.GetArrayFromImage(vol_sitk).astype(np.float32)  # (H, W, Z)
    
    # Add logging to check dimensions
    logging.info(f"Original volume shape: {vol_np.shape}")
    
    # Transpose from (H, W, Z) to (Z, H, W)
    vol_np = np.transpose(vol_np, (2, 0, 1))  # Reorder dimensions to (Z, H, W)
    logging.info(f"Volume shape after transpose: {vol_np.shape}")
    
    # Ensure correct dimensions (Z, H, W) -> (1, Z, H, W)
    if len(vol_np.shape) == 3:
        vol_np = np.expand_dims(vol_np, axis=0)  # Add channel dimension
    logging.info(f"Volume shape after adding channel: {vol_np.shape}")
    
    # Apply preprocessing transforms
    preprocessed = pre_seg_transforms(vol_np)
    logging.info(f"Volume shape after preprocessing: {preprocessed.shape}")
    
    # Convert MetaTensor to numpy array first, then to torch tensor
    if hasattr(preprocessed, 'numpy'):
        preprocessed = preprocessed.numpy()
    vol_tensor = torch.from_numpy(preprocessed).cuda()
    # Add batch dimension for 5D input (batch, channel, depth, height, width)
    vol_tensor = vol_tensor.unsqueeze(0)  # Add batch dimension
    logging.info(f"Final tensor shape: {vol_tensor.shape}")

    # Step 2: Inference segmentation
    logging.info("Step 2: Performing segmentation")
    with torch.no_grad():
        # Log input tensor properties
        logging.info(f"Input tensor shape: {vol_tensor.shape}")
        logging.info(f"Input tensor range: [{vol_tensor.min():.2f}, {vol_tensor.max():.2f}]")
        
        # Use the same ROI size as in the model config
        roi_size = (96, 96, 96)  # 3D ROI size as specified in the model config
        logging.info(f"Using ROI size: {roi_size}")
        
        # Perform sliding window inference with model's recommended settings
        seg_output = sliding_window_inference(
            vol_tensor, 
            roi_size=roi_size, 
            sw_batch_size=4,  # As specified in model config
            predictor=model,
            overlap=0.625,    # As specified in model config
            mode="gaussian",  # Use Gaussian weighting for smoother boundaries
            sw_device=torch.device("cuda")
        )
        
        # Apply postprocessing as specified in model config
        seg_output = torch.softmax(seg_output, dim=1)  # Apply softmax
        seg_mask = torch.argmax(seg_output, dim=1).squeeze().cpu().numpy()
        
        # Verify segmentation mask
        logging.info(f"Segmentation mask shape: {seg_mask.shape}")
        logging.info(f"Unique labels in segmentation: {np.unique(seg_mask)}")
        logging.info(f"Label counts: {np.bincount(seg_mask.flatten())}")
        
        # Transpose segmentation mask to match original volume orientation (H, W, Z)
        seg_mask = np.transpose(seg_mask, (1, 2, 0))  # From (Z, H, W) to (H, W, Z)
        logging.info(f"Transposed segmentation mask shape: {seg_mask.shape}")
        
        # Save raw segmentation output for inspection
        raw_seg = seg_output[0, 0].cpu().numpy()
        raw_seg = np.transpose(raw_seg, (1, 2, 0))  # From (Z, H, W) to (H, W, Z)
        raw_seg_save_name = f"{study_uid}_{series_uid}_raw_segmentation"
        save_image(raw_seg, SEGMENTATION_DIR, raw_seg_save_name, "volume", vol_resampled)
        
        # Save segmentation mask
        seg_save_name = f"{study_uid}_{series_uid}_segmentation"
        save_image(seg_mask, SEGMENTATION_DIR, seg_save_name, "segmentation", vol_resampled)
        
        # Save overlay of segmentation on resampled volume
        # Convert both images to float32 before creating overlay
        vol_array = sitk.GetArrayFromImage(vol_resampled).astype(np.float32)
        seg_array = seg_mask.astype(np.float32)
        overlay_array = vol_array * 0.7 + seg_array * 0.3
        overlay_image = sitk.GetImageFromArray(overlay_array)
        overlay_image.CopyInformation(vol_resampled)
        overlay_save_name = f"{study_uid}_{series_uid}_overlay"
        save_image(overlay_image, SEGMENTATION_DIR, overlay_save_name, "volume", vol_resampled)

    # Step 3: Crop by organ bounds
    logging.info("Step 3: Cropping volume")
    zmin, zmax = get_crop_bounds(seg_mask, INCLUDED_LABELS, z_margin=Z_MARGIN)
    logging.info(f"Crop bounds - z_min: {zmin}, z_max: {zmax}, crop size: {zmax - zmin}")
    
    # Verify crop bounds
    if zmax <= zmin:
        logging.error(f"Invalid crop bounds: z_min={zmin}, z_max={zmax}")
        continue
        
    # Crop both the segmentation mask and the resampled volume
    cropped_seg = seg_mask[zmin:zmax]
    cropped_vol = sitk.GetArrayFromImage(vol_resampled)[zmin:zmax]
    
    # Save cropped segmentation
    cropped_seg_save_name = f"{study_uid}_{series_uid}_cropped_segmentation"
    save_image(cropped_seg, CROPPED_DIR, cropped_seg_save_name, "segmentation", vol_resampled)
    
    # Save cropped volume
    cropped_vol_save_name = f"{study_uid}_{series_uid}_cropped_volume"
    save_image(cropped_vol, CROPPED_DIR, cropped_vol_save_name, "volume", vol_resampled)
    
    # Create and save cropped overlay
    cropped_vol_array = cropped_vol.astype(np.float32)
    cropped_seg_array = cropped_seg.astype(np.float32)
    cropped_overlay_array = cropped_vol_array * 0.7 + cropped_seg_array * 0.3
    cropped_overlay_image = sitk.GetImageFromArray(cropped_overlay_array)
    cropped_overlay_image.CopyInformation(vol_resampled)
    cropped_overlay_save_name = f"{study_uid}_{series_uid}_cropped_overlay"
    save_image(cropped_overlay_image, CROPPED_DIR, cropped_overlay_save_name, "volume", vol_resampled)
    
    slice_counts.append(zmax - zmin)

# Step 5: Compute average cropped length and resample
avg_slices = int(np.round(np.mean(slice_counts)))
logging.info(f"Average cropped length: {avg_slices} slices")

# Optional: Re-resample all saved volumes to avg_slices
logging.info("Step 5: Final resampling to average length")
for fname in os.listdir(CROPPED_DIR):
    img_path = os.path.join(CROPPED_DIR, fname)
    img = sitk.ReadImage(img_path)
    original_size = img.GetSize()
    logging.info(f"Resampling {fname} from size {original_size} to {avg_slices} slices")
    img_resampled = resample_to_k_slices(img, avg_slices)
    sitk.WriteImage(img_resampled, img_path)
    logging.info(f"Final size after resampling: {img_resampled.GetSize()}")

logging.info("Preprocessing completed successfully")
