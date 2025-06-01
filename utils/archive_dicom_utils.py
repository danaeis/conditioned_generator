import os
import pydicom
import numpy as np
import SimpleITK as sitk

def is_valid_dicom(dcm):
    try:
        # Check for essential attributes like ImagePositionPatient and PixelData
        return hasattr(dcm, 'ImagePositionPatient') and hasattr(dcm, 'PixelData')
    except Exception:
        return False


def load_dicom_volume_with_metadata(dicom_path):
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(dicom_path)
    if not series_ids:
        raise FileNotFoundError(f"No DICOM series found in {dicom_path}")
    
    series_file_names = reader.GetGDCMSeriesFileNames(dicom_path, series_ids[0])
    reader.SetFileNames(series_file_names)
    image = reader.Execute()

    # Extract metadata from the first file
    metadata_dict = {}
    for key in reader.GetMetaDataKeys(0):
        metadata_dict[key] = reader.GetMetaData(0, key)

    return image, metadata_dict







def get_dicom_metadata(dcm):
    """Extract relevant metadata from a single DICOM slice."""
    metadata = {}
    try:
        # Extract display window information
        metadata['WindowCenter'] = float(getattr(dcm, 'WindowCenter', None)) if getattr(dcm, 'WindowCenter', None) is not None else None
        metadata['WindowWidth'] = float(getattr(dcm, 'WindowWidth', None)) if getattr(dcm, 'WindowWidth', None) is not None else None

        # Extract rescaling information
        metadata['RescaleSlope'] = float(getattr(dcm, 'RescaleSlope', 1.0))
        metadata['RescaleIntercept'] = float(getattr(dcm, 'RescaleIntercept', 0.0))

        # Extract photometric interpretation
        metadata['PhotometricInterpretation'] = getattr(dcm, 'PhotometricInterpretation', 'MONOCHROME2')

        # Extract patient/study/series information
        metadata['StudyInstanceUID'] = getattr(dcm, 'StudyInstanceUID', None)
        metadata['SeriesInstanceUID'] = getattr(dcm, 'SeriesInstanceUID', None)
        metadata['SOPInstanceUID'] = getattr(dcm, 'SOPInstanceUID', None)
        metadata['Modality'] = getattr(dcm, 'Modality', None)
        metadata['PatientID'] = getattr(dcm, 'PatientID', None)
        metadata['StudyID'] = getattr(dcm, 'StudyID', None)
        metadata['SeriesNumber'] = getattr(dcm, 'SeriesNumber', None)

    except Exception as e:
        print(f"Warning: Could not extract some metadata from DICOM slice: {e}")

    # SimpleITK stores metadata as strings, so ensure all values are strings
    return {k: str(v) if v is not None else "" for k, v in metadata.items()}

def load_dicom_series(series_path):
    """Load a DICOM series into a 3D volume with spacing, origin, direction, and metadata."""
    slice_files = [os.path.join(series_path, f) for f in os.listdir(series_path) if f.endswith('.dcm')]
    slices = [pydicom.dcmread(f) for f in slice_files]

    # Filter out invalid DICOMs that lack essential attributes
    valid_slices = [s for s in slices if is_valid_dicom(s)]
    if not valid_slices:
        raise ValueError(f"No valid DICOM slices found in {series_path}")

    # Sort slices by their ImagePositionPatient[2] (z-coordinate)
    # We sort based on the z-position relative to the first slice to handle potential variations
    # Ensure there is at least one slice before sorting based on it
    try:
        valid_slices.sort(key=lambda s: float(valid_slices[0].ImagePositionPatient[2] - s.ImagePositionPatient[2]))
    except IndexError:
        # Handle case where valid_slices has only one element
        pass # Sorting is not needed for a single slice

    # Get metadata from the first valid slice. Assume metadata is consistent across series.
    # Ensure there is at least one slice before accessing index 0
    if not valid_slices:
         raise ValueError(f"No valid slices after sorting attempt in {series_path}")
    metadata = get_dicom_metadata(valid_slices[0])

    # Build 3D numpy array from pixel data
    image_3d = np.stack([s.pixel_array for s in valid_slices], axis=-1)

    # Apply rescale slope and intercept to get true pixel values (in Hounsfield Units for CT)
    slope = float(metadata.get('RescaleSlope', 1.0))
    intercept = float(metadata.get('RescaleIntercept', 0.0))
    if slope != 1.0 or intercept != 0.0:
         image_3d = image_3d * slope + intercept

    # Convert to int16 to preserve typical CT data range and precision
    image_3d = image_3d.astype(np.int16)

    # Get spacing (Pixel Spacing for xy, calculated difference for z)
    spacing_xy = [float(x) for x in valid_slices[0].PixelSpacing]  # x, y spacing
    try:
        # Calculate z-spacing from difference in ImagePositionPatient[2]
        # Ensure there are at least two slices before calculating spacing this way
        if len(valid_slices) >= 2:
            spacing_z = float(valid_slices[1].ImagePositionPatient[2] - valid_slices[0].ImagePositionPatient[2])
            # Ensure positive spacing
            if spacing_z < 0:
                spacing_z = -spacing_z
                # If spacing is negative, the slices might be sorted in reverse order. Reversing the array.
                image_3d = image_3d[:, :, ::-1]
        else:
             # Fallback to SliceThickness if less than 2 slices
             spacing_z = float(getattr(valid_slices[0], 'SliceThickness', 1.0))
             print(f"Warning: Using SliceThickness ({spacing_z}) for z-spacing as less than 2 slices found in {series_path}")

    except (ValueError, TypeError) as e:
        # Fallback to SliceThickness for other errors during spacing calculation
        spacing_z = float(getattr(valid_slices[0], 'SliceThickness', 1.0)) # Fallback
        print(f"Warning: Using SliceThickness ({spacing_z}) for z-spacing due to error calculating from ImagePositionPatient: {e} in {series_path}")

    spacing = (spacing_xy[0], spacing_xy[1], spacing_z)

    # Get origin (ImagePositionPatient of the first slice)
    # SimpleITK expects origin as a tuple of floats
    origin = tuple(float(o) for o in valid_slices[0].ImagePositionPatient)

    # Get direction cosine matrix
    try:
        # ImageOrientationPatient (6 elements) gives direction cosines of row and column vectors
        # [Xx, Xy, Xz, Yx, Yy, Yz]
        ior = [float(x) for x in valid_slices[0].ImageOrientationPatient]
        row_cos = np.array(ior[:3])
        col_cos = np.array(ior[3:])
        # The slice normal (z-direction) is the cross product of row and column cosines
        slice_cos = np.cross(row_cos, col_cos)
        # SimpleITK direction is a 9-element tuple (row x, row y, row z, col x, col y, col z, slice x, slice y, slice z)
        direction = tuple(row_cos.tolist() + col_cos.tolist() + slice_cos.tolist())
    except (AttributeError, ValueError, TypeError) as e:
        print(f"Warning: Could not derive ImageOrientationPatient direction, using identity: {e} in {series_path}")
        # Fallback to identity matrix if ImageOrientationPatient is missing or invalid
        direction = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

    # Create SimpleITK image
    # SimpleITK expects array in Z, Y, X order, so transpose the numpy array
    # Ensure image_3d is 3D; if only one slice, np.stack might result in 2D
    if image_3d.ndim == 2:
        image_3d = image_3d[:, :, np.newaxis] # Add a z-dimension for single slices
    sitk_image = sitk.GetImageFromArray(image_3d)

    # Set spatial information
    sitk_image.SetSpacing(spacing)
    sitk_image.SetOrigin(origin)
    sitk_image.SetDirection(direction)

    return sitk_image, metadata
