import numpy as np
import logging
import os
import SimpleITK as sitk


def crop_abdomen_region(image, lower_hu=-200, upper_hu=300, margin=10):
    """
    Crops the abdomen region from a 3D image numpy array based on HU range.
    
    Args:
        image (np.ndarray): 3D numpy array (depth, height, width).
        lower_hu (int): Lower Hounsfield Unit threshold.
        upper_hu (int): Upper Hounsfield Unit threshold.
        margin (int): Margin to add around the detected region.
        
    Returns:
        tuple: A tuple containing:
            - np.ndarray: The cropped image array.
            - tuple: The cropping bounds as (min_coords, max_coords).
    """
    # The input image array is expected to be in SimpleITK's numpy order (Z, Y, X)
    # coords will be (z, y, x)
    mask = (image > lower_hu) & (image < upper_hu)
    coords = np.argwhere(mask)
    if coords.size == 0:
        # If no abdomen region found, return the original image and bounds covering the whole image
        logging.warning("No abdomen region found based on HU range. Returning original volume and full bounds.")
        min_coords = np.array([0, 0, 0])
        max_coords = np.array(image.shape) # shape is (Z, Y, X)
        return image, (min_coords, max_coords)

    min_coords = np.maximum(coords.min(axis=0) - margin, 0)
    # max_coords should be inclusive of the last element in the slice, so no +1 here
    max_coords = np.minimum(coords.max(axis=0) + margin, image.shape - 1)

    slices = tuple(slice(min_c, max_c + 1) for min_c, max_c in zip(min_coords, max_coords))
    
    cropped_array = image[slices]
    
    # Return the cropped array and the calculated bounds (min_coords, max_coords)
    return cropped_array, (min_coords, max_coords)


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



def save_original_volume(image, save_dir, filename):
    """
    Save SimpleITK image as NIfTI (.nii.gz) preserving original quality.
    
    Args:
        image (SimpleITK.Image): The image to save.
        save_dir (str): Directory to save into.
        filename (str): File name (without extension).
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{filename}.nii.gz")
    
    try:
        # Save without changing type, spacing, etc.
        sitk.WriteImage(image, save_path, useCompression=True)
        logging.info(f"Saved volume at native quality to {save_path}")
    except Exception as e:
        logging.error(f"Could not save volume {filename}: {e}")
        return None

    return save_path


def save_image(image, save_dir, filename, reference_image=None, metadata=None):
    os.makedirs(save_dir, exist_ok=True)

    # Ensure image is a SimpleITK image
    if isinstance(image, np.ndarray):
        # Default: float32 to preserve dynamic range (e.g., Hounsfield Units)
        sitk_image = sitk.GetImageFromArray(image.astype(np.float32))
    else:
        sitk_image = image

    # Preserve spacing/origin/direction if provided
    if reference_image is not None:
        try:
            sitk_image.CopyInformation(reference_image)
        except Exception as e:
            logging.warning(f"CopyInformation failed: {e}")
    
    # Save image in compressed NIfTI format
    save_path = os.path.join(save_dir, f"{filename}.nii.gz")
    try:
        sitk.WriteImage(sitk_image, save_path, useCompression=True)
    except Exception as e:
        logging.error(f"Failed to save: {e}")
        return None

    return save_path


# def save_image(image, save_dir, filename, image_type="volume", reference_image=None, metadata=None):
#     """
#     Save an image in NIfTI format with proper metadata.
    
#     Args:
#         image: SimpleITK image or numpy array to save
#         save_dir: Directory to save the image
#         filename: Name of the file (without extension)
#         image_type: Type of image ("volume" or "segmentation")
#         reference_image: Reference image to copy spatial information from (optional)
#         metadata: Dictionary containing DICOM metadata to add to NIfTI header (optional)
#     """
#     os.makedirs(save_dir, exist_ok=True)
    
#     # Convert numpy array to SimpleITK image if needed, maintaining appropriate data type
#     if isinstance(image, np.ndarray):
#         if image_type == "segmentation":
#             # Segmentation masks should be unsigned integers
#             sitk_image = sitk.GetImageFromArray(image.astype(np.uint8))
#         elif image_type == "volume":
#             # Volume data (like CT) is typically signed 16-bit integers
#             sitk_image = sitk.GetImageFromArray(image.astype(np.int16))
#         else:
#              # Default to float32 for other image types
#              sitk_image = sitk.GetImageFromArray(image.astype(np.float32))
#     else:
#         # If input is already a SimpleITK image, use it directly
#         sitk_image = image
    
#     # Copy spatial information (spacing, origin, direction) from reference image if provided
#     # This is important if the input 'image' is a numpy array derived from a SimpleITK image
#     if reference_image is not None:
#         try:
#             sitk_image.CopyInformation(reference_image)
#         except Exception as e:
#             logging.warning(f"Could not copy information from reference image: {e}")
#             # Attempt to set information manually if CopyInformation fails
#             try:
#                 sitk_image.SetSpacing(reference_image.GetSpacing())
#                 sitk_image.SetOrigin(reference_image.GetOrigin())
#                 sitk_image.SetDirection(reference_image.GetDirection())
#             except Exception as e_manual:
#                  logging.warning(f"Could not manually set information from reference image: {e_manual}")


    
#     # Add metadata to image header if provided
#     if metadata is not None:
#         for key, value in metadata.items():
#             # SimpleITK stores metadata as strings in the image header
#             # Ensure key is not empty and value is convertible to string
#             if key and value is not None:
#                 try:
#                     sitk_image.SetMetaData(str(key), str(value))
#                 except Exception as e:
#                     logging.warning(f"Could not set metadata key '{key}' with value '{value}': {e}")

    
#     # Create full path and save
#     save_path = os.path.join(save_dir, f"{filename}.nii.gz")
    
#     # Use compression for smaller file size (default compression level is fine)
#     try:
#         sitk.WriteImage(sitk_image, save_path, useCompression=True)
#         logging.info(f"Saved {image_type} to: {save_path}")
#     except Exception as e:
#         logging.error(f"Failed to save image {save_path}: {e}")
#         return None # Indicate failure

#     return save_path



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


