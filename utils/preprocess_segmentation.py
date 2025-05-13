import os
import pandas as pd
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import json

import torch
import nibabel as nib
from monai.transforms import Orientation, Spacing, ScaleIntensity
from transformers import AutoModelForImageSegmentation
import monai
from huggingface_hub import hf_hub_download
from monai.bundle import ConfigParser
from monai.networks.nets import SwinUNETR
import torch
from monai.networks.nets import UNETR

from dicom_utils import load_dicom_series
from volume_utils import crop_center, resample_to_target, resample_spacing, detect_abdomen_bbox
from registration_utils import register_to_atlas, save_registered_image
from load_and_cache_dicom_series import load_and_cache_dicom_series


def save_abdomen_bounds(seg_mask, sitk_img, dicom_series_dir, included_labels, z_margin=5):
    """
    Save slice range for specified label(s) into a JSON file.
    
    Args:
        seg_mask (np.ndarray): 3D array [H, W, Z]
        sitk_img (SimpleITK.Image): Original volume (for metadata if needed)
        dicom_series_dir (str): Path to the SeriesInstanceUID folder
        included_labels (set): e.g., {1, 2, 3, 4} (use BTCV label IDs for organs)
        z_margin (int): Margin slices above and below the bounding box
    """
    # Convert to [Z, H, W] for axis operations
    seg_mask = np.transpose(seg_mask, (2, 0, 1))  # (Z, H, W)
    mask = np.isin(seg_mask, list(included_labels))  # Boolean mask of relevant labels

    z_mask = np.any(mask, axis=(1, 2))  # Along H and W
    if not np.any(z_mask):
        z_min, z_max = 0, seg_mask.shape[0] - 1
    else:
        z_indices = np.where(z_mask)[0]
        z_min = max(z_indices[0] - z_margin, 0)
        z_max = min(z_indices[-1] + z_margin, seg_mask.shape[0] - 1)

    # Save to same directory as DICOM series
    output_path = os.path.join(dicom_series_dir, "abdominal_bounds.json")
    with open(output_path, "w") as f:
        json.dump({"z_min": int(z_min), "z_max": int(z_max)}, f, indent=2)


def resample_to_250_slices(volume, original_spacing, target_slices=250):
    size = volume.GetSize()
    spacing = list(original_spacing)
    new_spacing = list(spacing)
    new_spacing[2] = spacing[2] * size[2] / target_slices
    return resample_spacing(volume, tuple(new_spacing))

def crop_abdomen_from_seg(seg_mask, z_margin=5):
    z_mask = np.any(seg_mask > 0, axis=(0, 1))
    if not np.any(z_mask):
        return 0, seg_mask.shape[2]  # fallback to full volume
    z_indices = np.where(z_mask)[0]
    z_min = max(z_indices[0] - z_margin, 0)
    z_max = min(z_indices[-1] + z_margin + 1, seg_mask.shape[2])
    return z_min, z_max


# --- CONFIG ---
BATCH_DIR = '../../ncct_cect/vindr_ds/batches'
LABELS_CSV = '../../ncct_cect/vindr_ds/labels.csv'
ALIGNED_SLICES_DIR = '../../ncct_cect/vindr_ds/aligned_slices'
REGISTERED_DIR = '../../ncct_cect/vindr_ds/registered_volumes'
ATLAS_SAVE_PATH = '../../ncct_cect/vindr_ds/atlas/reference.nii.gz'
CACHE_PATH = '../../ncct_cect/vindr_ds/cached_dicom_series.pkl'

STANDARD_SPACING = (1.0, 1.0, 1.0)
FIXED_SHAPE = (192, 192, 64)
TRANSFORM_TYPE = 'rigid'

dicom_series, z_spacings = load_and_cache_dicom_series(BATCH_DIR, LABELS_CSV, CACHE_PATH)
print("dicom series loaded")



# Define preprocessing (must match bundle settings)
pre_seg_transforms = monai.transforms.Compose([
    Orientation(axcodes="RAS"),
    ScaleIntensity(),
    monai.transforms.EnsureChannelFirst(),
    Spacing(pixdim=(1.0, 1.0, 1.0), mode="bilinear")
])

model = UNETR(
    in_channels=1,
    out_channels=14,
    img_size=(96, 96, 96),
    feature_size=16,
    hidden_size=768,
    mlp_dim=3072,
    num_heads=12,
    #pos_embed='perceptron',
    norm_name='instance',
    res_block=True,
    dropout_rate=0.0,
)

model.load_state_dict(torch.load("unetr_model/UNETR_model_best_acc.pth"))
model.eval().cuda()

from pathlib import Path

# Set of label IDs to include for bounding box (BTCV or AMOS label IDs)
included_labels = {1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13}  # Exclude arteries (e.g. 4)

for study_uid, series_uid, volume in tqdm(dicom_series):
    spacing = volume.GetSpacing()

    # Step 1: Resample to 250 slices
    vol_250 = resample_to_250_slices(volume, spacing)

    # Step 2: Preprocess image for model
    vol_np = sitk.GetArrayFromImage(vol_250).astype(np.float32)  # (Z, H, W)
    vol_np = np.transpose(vol_np, (1, 2, 0))  # (H, W, Z)
    vol_tensor = pre_seg_transforms(vol_np).unsqueeze(0).cuda()  # (1, 1, H, W, Z)

    # Step 3: Run segmentation
    with torch.no_grad():
        output = model(vol_tensor)
        seg_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()  # (H, W, Z)

    # Step 4: Locate SeriesInstanceUID folder
    dicom_series_dir = os.path.join(BATCH_DIR, study_uid, series_uid)
    if not os.path.exists(dicom_series_dir):
        print(f"Warning: path {dicom_series_dir} not found. Skipping save.")
        continue

    # Step 5: Save slice bounds as JSON
    save_abdomen_bounds(
        seg_mask=seg_mask,
        sitk_img=vol_250,
        dicom_series_dir=dicom_series_dir,
        included_labels=included_labels,
        z_margin=5,
    )

    # Step 6: Load saved bounds and crop image accordingly
    bounds_path = os.path.join(dicom_series_dir, "abdominal_bounds.json")
    with open(bounds_path, "r") as f:
        bounds = json.load(f)
        zmin = bounds["z_min"]
        zmax = bounds["z_max"]

    # Step 7: Crop and save cropped volume
    cropped_array = sitk.GetArrayFromImage(vol_250)[zmin:zmax, :, :]
    cropped_image = sitk.GetImageFromArray(cropped_array)
    cropped_image.SetSpacing(vol_250.GetSpacing())
    cropped_image.SetOrigin(vol_250.GetOrigin())
    cropped_image.SetDirection(vol_250.GetDirection())

    # Save in ALIGNED_SLICES_DIR
    save_name = f"{study_uid}_{series_uid}_abdomen_cropped.nii.gz"
    save_path = os.path.join(ALIGNED_SLICES_DIR, save_name)
    sitk.WriteImage(cropped_image, save_path)
