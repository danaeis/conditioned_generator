import numpy as np
import logging
import os
import SimpleITK as sitk

def crop_abdomen_region(image, lower_hu=-200, upper_hu=300, margin=10):
    mask = (image > lower_hu) & (image < upper_hu)
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise ValueError("No abdomen region found based on HU range.")

    min_coords = np.maximum(coords.min(axis=0) - margin, 0)
    max_coords = np.minimum(coords.max(axis=0) + margin + 1, image.shape)

    slices = tuple(slice(min_c, max_c) for min_c, max_c in zip(min_coords, max_coords))
    return image[slices]


def resample_to_k_slices(image, k):
    """
    Resample a 3D image to have k slices while maintaining width x height dimensions.
    Args:
        image: SimpleITK image
        k: target number of slices
    Returns:
        Resampled SimpleITK image
    """
    # Get original image properties
    original_size = image.GetSize()  # (depth, width, height)
    original_spacing = image.GetSpacing()  # (depth, width, height)
    original_direction = image.GetDirection()
    original_origin = image.GetOrigin()
    
    logging.info(f"Original volume size (depth, width, height): {original_size}")
    logging.info(f"Original spacing (depth, width, height): {original_spacing}")
    
    # Calculate new spacing to maintain aspect ratio
    new_spacing = list(original_spacing)
    new_spacing[0] = (original_size[0] * original_spacing[0]) / k  # adjust depth spacing
    
    # Set new size (change depth to k, keep width and height the same)
    new_size = [k, original_size[1], original_size[2]]  # (depth, width, height)
    
    # Create and configure resampling filter
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(tuple(new_spacing))
    resample.SetSize(new_size)
    resample.SetOutputDirection(original_direction)
    resample.SetOutputOrigin(original_origin)
    resample.SetInterpolator(sitk.sitkLinear)
    
    resampled_image = resample.Execute(image)
    
    # Log resampling results
    output_size = resampled_image.GetSize()
    output_spacing = resampled_image.GetSpacing()
    logging.info(f"After resampling - Size (depth, width, height): {output_size}")
    logging.info(f"After resampling - Spacing (depth, width, height): {output_spacing}")
    
    return resampled_image

def get_crop_bounds(seg_mask, included_labels, z_margin=5):
    # Create a boolean array where each slice is True if it contains any of the included labels
    z_slices = np.any(np.isin(seg_mask, list(included_labels)), axis=(1, 2))
    indices = np.where(z_slices)[0]
    if len(indices) == 0:
        # If no slice contains any of the included labels, keep the whole volume
        logging.warning("No slices found containing included labels, keeping whole volume")
        return 0, seg_mask.shape[0]
    # Determine the topmost and bottommost slice that contains at least one of the included labels
    z_min = max(indices[0] - z_margin, 0)
    z_max = min(indices[-1] + z_margin + 1, seg_mask.shape[0])  # +1 to include the last slice
    logging.info(f"Crop bounds - z_min: {z_min}, z_max: {z_max}, crop size: {z_max - z_min}")
    return z_min, z_max

def save_image(image, save_dir, filename, image_type="volume", reference_image=None):
    """
    Save an image in NIfTI format with proper metadata.
    
    Args:
        image: SimpleITK image or numpy array to save
        save_dir: Directory to save the image
        filename: Name of the file (without extension)
        image_type: Type of image ("volume" or "segmentation")
        reference_image: Reference image to copy metadata from (optional)
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert numpy array to SimpleITK image if needed
    if isinstance(image, np.ndarray):
        if image_type == "segmentation":
            image = sitk.GetImageFromArray(image.astype(np.uint8))
        else:
            image = sitk.GetImageFromArray(image.astype(np.float32))
    
    # Copy metadata from reference image if provided
    if reference_image is not None:
        image.CopyInformation(reference_image)
    
    # Create full path and save
    save_path = os.path.join(save_dir, f"{filename}.nii.gz")
    sitk.WriteImage(image, save_path)
    logging.info(f"Saved {image_type} to: {save_path}")
    return save_path



# def resample_to_fixed_depth(image: sitk.Image, target_slices=128):
#     """Resample image to a fixed number of slices (depth) along Z-axis."""
#     original_spacing = image.GetSpacing()
#     size = image.GetSize()
#     new_spacing = list(original_spacing)
#     new_spacing[2] = original_spacing[2] * size[2] / target_slices
#     return resample_spacing(image, tuple(new_spacing))


def extract_abdominal_roi_slices(seg_mask: np.ndarray, organs: set, z_margin: int = 5):
    """
    Extract Z-slice bounds covering specified organs from a 3D segmentation mask.

    Args:
        seg_mask (np.ndarray): Segmentation mask of shape (H, W, Z)
        organs (set): Set of integer organ labels (e.g., {1, 2, 3})
        z_margin (int): Number of slices to expand above and below the ROI

    Returns:
        z_min (int), z_max (int): Cropping bounds [inclusive, exclusive)
    """
    assert seg_mask.ndim == 3, "Expected 3D segmentation mask (H, W, Z)"

    # Transpose to (Z, H, W) for easier Z-axis indexing
    seg_mask_z_first = np.transpose(seg_mask, (2, 0, 1))  # (Z, H, W)
    
    # Binary mask of where any specified organ appears
    organ_mask = np.isin(seg_mask_z_first, list(organs))  # (Z, H, W) boolean

    # Determine which slices have any organ
    z_any = np.any(organ_mask, axis=(1, 2))  # (Z,)
    if not np.any(z_any):
        return 0, seg_mask.shape[2]  # No organs found; return full range

    z_indices = np.where(z_any)[0]
    z_min = max(z_indices[0] - z_margin, 0)
    z_max = min(z_indices[-1] + z_margin + 1, seg_mask.shape[2])  # +1 for exclusive bound

    return z_min, z_max


