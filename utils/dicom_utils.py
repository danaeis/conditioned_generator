import os
import pydicom
import numpy as np
import SimpleITK as sitk

def is_valid_dicom(dcm):
    try:
        return hasattr(dcm, 'ImagePositionPatient') and hasattr(dcm, 'PixelData')
    except:
        return False

def load_dicom_series(series_path):
    """Load a DICOM series into a 3D volume with spacing and origin."""
    slice_files = [os.path.join(series_path, f) for f in os.listdir(series_path) if f.endswith('.dcm')]
    slices = [pydicom.dcmread(f) for f in slice_files]

    # Filter out invalid DICOMs
    slices = [s for s in slices if is_valid_dicom(s)]
    if len(slices) == 0:
        raise ValueError(f"No valid DICOM slices found in {series_path}")

    # Sort slices by z-axis position
    slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))

    # Build 3D array
    image_3d = np.stack([s.pixel_array for s in slices], axis=-1)
    image_3d = image_3d.astype(np.int16)  # Ensure signed 16-bit

    # Get spacing
    spacing_xy = [float(x) for x in slices[0].PixelSpacing]  # x, y
    try:
        spacing_z = float(slices[1].ImagePositionPatient[2] - slices[0].ImagePositionPatient[2])
    except:
        spacing_z = float(slices[0].SliceThickness)  # fallback
    spacing = (spacing_xy[0], spacing_xy[1], spacing_z)

    origin = slices[0].ImagePositionPatient
    direction = np.eye(3).flatten().tolist()  # Assume identity direction

    return image_3d, spacing, origin, direction
