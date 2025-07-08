import numpy as np
import logging
import os
import SimpleITK as sitk
import nibabel as nib

from scipy.ndimage import label, binary_fill_holes, binary_erosion, binary_dilation


# def resample_to_k_slices(image, k):
#     """
#     Resample a 3D image to have k slices while maintaining width x height dimensions.
#     Args:
#         image: SimpleITK image
#         k: target number of slices
#     Returns:
#         Resampled SimpleITK image
#     """
#     # Get original image properties
#     original_size = image.GetSize()  # (depth, width, height)
#     original_spacing = image.GetSpacing()  # (depth, width, height)
#     original_direction = image.GetDirection()
#     original_origin = image.GetOrigin()
    
#     logging.info(f"Original volume size (depth, width, height): {original_size}")
#     logging.info(f"Original spacing (depth, width, height): {original_spacing}")
    
#     # Calculate new spacing to maintain aspect ratio
#     new_spacing = list(original_spacing)
#     new_spacing[0] = (original_size[0] * original_spacing[0]) / k  # adjust depth spacing
    
#     # Set new size (change depth to k, keep width and height the same)
#     new_size = [k, original_size[1], original_size[2]]  # (depth, width, height)
    
#     # Create and configure resampling filter
#     resample = sitk.ResampleImageFilter()
#     resample.SetOutputSpacing(tuple(new_spacing))
#     resample.SetSize(new_size)
#     resample.SetOutputDirection(original_direction)
#     resample.SetOutputOrigin(original_origin)
#     resample.SetInterpolator(sitk.sitkLinear)
    
#     resampled_image = resample.Execute(image)
    
#     # Log resampling results
#     output_size = resampled_image.GetSize()
#     output_spacing = resampled_image.GetSpacing()
#     logging.info(f"After resampling - Size (depth, width, height): {output_size}")
#     logging.info(f"After resampling - Spacing (depth, width, height): {output_spacing}")
    
#     return resampled_image

# def get_crop_bounds(seg_mask, included_labels, z_margin=5):
#     # Create a boolean array where each slice is True if it contains any of the included labels
#     z_slices = np.any(np.isin(seg_mask, list(included_labels)), axis=(1, 2))
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



# def save_original_volume(image, save_dir, filename):
#     """
#     Save SimpleITK image as NIfTI (.nii.gz) preserving original quality.
    
#     Args:
#         image (SimpleITK.Image): The image to save.
#         save_dir (str): Directory to save into.
#         filename (str): File name (without extension).
#     """
#     os.makedirs(save_dir, exist_ok=True)
#     save_path = os.path.join(save_dir, f"{filename}.nii.gz")
    
#     try:
#         # Save without changing type, spacing, etc.
#         sitk.WriteImage(image, save_path, useCompression=True)
#         logging.info(f"Saved volume at native quality to {save_path}")
#     except Exception as e:
#         logging.error(f"Could not save volume {filename}: {e}")
#         return None

#     return save_path


# def save_image(image, save_dir, filename, reference_image=None, metadata=None):
#     os.makedirs(save_dir, exist_ok=True)

#     # Ensure image is a SimpleITK image
#     if isinstance(image, np.ndarray):
#         # Default: float32 to preserve dynamic range (e.g., Hounsfield Units)
#         sitk_image = sitk.GetImageFromArray(image.astype(np.float32))
#     else:
#         sitk_image = image

#     # Preserve spacing/origin/direction if provided
#     if reference_image is not None:
#         try:
#             sitk_image.CopyInformation(reference_image)
#         except Exception as e:
#             logging.warning(f"CopyInformation failed: {e}")
    
#     # Save image in compressed NIfTI format
#     save_path = os.path.join(save_dir, f"{filename}.nii.gz")
#     try:
#         sitk.WriteImage(sitk_image, save_path, useCompression=True)
#     except Exception as e:
#         logging.error(f"Failed to save: {e}")
#         return None

    # return save_path

def remove_ct_table(volume_np, threshold=-400, fill_value=-1000, padding=10):
    """Remove the CT table using thresholding and the largest component mask."""
    body_mask = volume_np > threshold
    labeled_array, num_features = label(body_mask)
    if num_features == 0:
        raise ValueError("No body detected. Try adjusting threshold.")
    
    # Get largest component
    largest_label = np.argmax(np.bincount(labeled_array.flat)[1:]) + 1
    body_mask = (labeled_array == largest_label)
    body_mask = binary_fill_holes(body_mask)
    
    if padding > 0:
        body_mask = binary_dilation(body_mask, iterations=padding)
    
    masked_volume = np.where(body_mask, volume_np, fill_value)
    return masked_volume

def get_bounding_box_from_mask(mask):
    """Get tight bounding box (as slices) from a binary mask."""
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    min_coords = coords.min(axis=0)
    max_coords = coords.max(axis=0) + 1
    return tuple(slice(min_, max_) for min_, max_ in zip(min_coords, max_coords))

def get_abdomen_bbox(volume_np, z_min=40, z_max=200, threshold=-300):
    """
    Get the bounding box of the abdomen based on axial projection.
    Only uses slices between z_min and z_max for speed.
    """
    subvol = volume_np[z_min:z_max]
    mask = subvol > threshold
    projection = np.any(mask, axis=0)  # [y, x]
    bbox_2d = get_bounding_box_from_mask(projection)
    if bbox_2d is None:
        raise ValueError("Could not find body region.")
    return (slice(z_min, z_max),) + bbox_2d

def symmetrically_extend_bbox(bbox, shape):
    """Symmetrically extend the bbox to balance cropping."""
    extended = []
    for sl, dim_len in zip(bbox, shape):
        start = sl.start
        stop = sl.stop
        pad = min(start, dim_len - stop)
        extended.append(slice(start, dim_len - pad))
    return tuple(extended)

def auto_detect_z_range(volume, threshold=-300):
    """Automatically find z range with non-zero tissue."""
    body_mask = volume > threshold
    non_zero_z = np.any(body_mask, axis=(1, 2))  # shape: [z]
    indices = np.where(non_zero_z)[0]
    if len(indices) == 0:
        raise ValueError("No tissue found.")
    return indices[0], indices[-1]

def get_body_mask_robust(volume_np):
    """
    Find the body mask using Otsu's thresholding and largest component.
    This is more robust than a fixed HU threshold.
    """
    from skimage.filters import threshold_otsu
    from scipy.ndimage import label, binary_fill_holes

    # Compute Otsu threshold on the whole volume
    flat = volume_np.flatten()
    # Remove extreme air values for Otsu
    flat = flat[flat > -900]
    otsu_thresh = threshold_otsu(flat)
    rough_mask = volume_np > otsu_thresh

    # Largest connected component
    labeled, num = label(rough_mask)
    if num == 0:
        raise ValueError("No body found in CT.")
    largest = (labeled == (np.bincount(labeled.flat)[1:].argmax() + 1))
    filled = binary_fill_holes(largest)
    return filled

def process_ct_and_crop_abdomen(volume_np, original_affine):
    """
    Process CT volume by robustly finding the body mask and focusing on abdomen region.
    """
    # Step 1: Find robust body mask
    body_mask = get_body_mask_robust(volume_np)

    # Step 2: Detect Z-range with actual body
    z_min, z_max = auto_detect_z_range(body_mask)
    if z_min >= z_max:
        raise ValueError("No valid z range found for abdomen cropping.")

    # Step 3: Compute abdomen bounding box from body mask in that z range
    bbox = get_abdomen_bbox(body_mask.astype(np.uint8), z_min=z_min, z_max=z_max, threshold=0)
    
    # Step 4: Create a mask for the abdomen region
    abdomen_mask = np.zeros_like(body_mask, dtype=bool)
    abdomen_mask[bbox] = True
    abdomen_mask &= body_mask  # Only keep inside the body

    # Step 5: Apply the mask to focus on abdomen while maintaining dimensions
    processed_np = np.where(abdomen_mask, volume_np, -1000)

    # Step 6: Create new NIfTI image with original affine
    processed_img = nib.Nifti1Image(processed_np, affine=original_affine)

    return processed_np, processed_img
