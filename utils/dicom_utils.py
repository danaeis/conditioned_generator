import os
import pickle
import pydicom
import pandas as pd
import SimpleITK as sitk
import nibabel as nib
import numpy as np
from typing import List, Dict, Optional
from collections import defaultdict

def get_slice_thickness(dicom_path: str) -> float:
    """Get slice thickness from DICOM series (in mm)."""
    try:
        first_file = next(f for f in os.listdir(dicom_path) if f.endswith('.dcm'))
        ds = pydicom.dcmread(os.path.join(dicom_path, first_file))
        return float(getattr(ds, 'SliceThickness', float('inf')))  # Use infinity as fallback
    except:
        return float('inf')  # Return infinity if error occurs

def save_dicom_paths(batch_dir: str, labels_csv: str, output_pkl: str) -> List[Dict]:
    """Select one optimal series per phase per study based on slice thickness."""
    labels_df = pd.read_csv(labels_csv).dropna(subset=['StudyInstanceUID', 'SeriesInstanceUID', 'Label'])
    phase_lookup = {
        (row['StudyInstanceUID'], row['SeriesInstanceUID']): row['Label'].lower()
        for _, row in labels_df.iterrows()
    }

    # {(study_uid, phase): (best_thickness, series_info)}
    best_series = defaultdict(lambda: (float('inf'), None))  # Initialize with infinity

    for batch in os.listdir(batch_dir):
        batch_path = os.path.join(batch_dir, batch)
        for study in os.listdir(batch_path):
            study_path = os.path.join(batch_path, study)
            for series in os.listdir(study_path):
                series_path = os.path.join(study_path, series)
                key = (study, series)
                
                if key not in phase_lookup:
                    continue
                
                phase = phase_lookup[key]
                thickness = get_slice_thickness(series_path)
                
                # Update if this series has finer slices than current best
                if thickness < best_series[(study, phase)][0]:
                    best_series[(study, phase)] = (
                        thickness,
                        {
                            'study_uid': study,
                            'series_uid': series,
                            'series_path': series_path,
                            'phase': phase,
                            'slice_thickness': thickness
                        }
                    )

    # Extract just the series info (without thickness tracking)
    final_series_data = [info for (thickness, info) in best_series.values()]
    
    with open(output_pkl, 'wb') as f:
        pickle.dump(final_series_data, f)
    
    print(f"Saved {len(final_series_data)} optimal series (1 per phase per study) to {output_pkl}")
    return final_series_data

def convert_dicom_to_nifti(
    dicom_path: str,
    output_path: str,
    reorient: bool = True,
    normalize: bool = True
) -> None:
    """Convert DICOM series to NIfTI format in BTCV-compliant structure."""
    # Collect and sort DICOM files based on ImagePositionPatient (Z axis)
    dicom_files = []
    for file in os.listdir(dicom_path):
        if not file.endswith(".dcm"):
            continue
        filepath = os.path.join(dicom_path, file)
        ds = pydicom.dcmread(filepath, stop_before_pixels=True)
        ipp = getattr(ds, "ImagePositionPatient", None)
        dicom_files.append((filepath, float(ipp[2]) if ipp else 0))

    # Sort by Z-axis (slice position)
    dicom_files.sort(key=lambda x: x[1])
    sorted_filepaths = [f[0] for f in dicom_files]

    # Read sorted DICOM series
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(sorted_filepaths)
    image = reader.Execute()

    # Reorient to RAS if specified (BTCV requires RAS orientation)
    if reorient:
        # First, get the current orientation
        current_orientation = sitk.DICOMOrientImageFilter.GetOrientationFromDirectionCosines(image.GetDirection())
        print(f"Original orientation: {current_orientation}")

        # Reorient to RAS
        orient_filter = sitk.DICOMOrientImageFilter()
        orient_filter.SetDesiredCoordinateOrientation("RAS")
        image = orient_filter.Execute(image)

        # Verify the orientation
        new_orientation = sitk.DICOMOrientImageFilter.GetOrientationFromDirectionCosines(image.GetDirection())
        print(f"New orientation: {new_orientation}")
        
        # Verify direction matrix
        direction = image.GetDirection()
        print(f"Direction matrix: {direction}")
        
        # Expected RAS direction matrix should be (1,0,0,0,1,0,0,0,1)
        expected_direction = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        if direction != expected_direction:
            print("Warning: Direction matrix is not in RAS orientation!")
            print(f"Expected: {expected_direction}")
            print(f"Got: {direction}")

    # Optional intensity normalization
    if normalize:
        array = sitk.GetArrayFromImage(image).astype(np.float32)
        array = (array - np.min(array)) / (np.max(array) - np.min(array))
        image = sitk.GetImageFromArray(array)
        image.CopyInformation(image)  # preserve spacing/origin/direction

    # Ensure float32 type for images (BTCV standard)
    image = sitk.Cast(image, sitk.sitkFloat32)

    # Save as NIfTI
    sitk.WriteImage(image, output_path)
    
    # Verify the saved image
    saved_image = sitk.ReadImage(output_path)
    saved_direction = saved_image.GetDirection()
    print(f"Saved image direction matrix: {saved_direction}")
    if saved_direction != expected_direction:
        print("Warning: Saved image is not in RAS orientation!")

def process_series(
    series_data: List,  # Accepts both tuples and dicts
    nifti_root_dir: str,
    overwrite: bool = False
) -> None:
    """
    Process DICOM series in either tuple or dictionary format.
    Tuple format: (study_uid, series_uid, sitk_image, metadata)
    Dict format: {'study_uid': ..., 'series_uid': ..., ...}
    """
    for series in series_data:
        try:
            # Handle both tuple and dictionary formats
            if isinstance(series, tuple):
                study_uid, series_uid, sitk_image, metadata = series
                dicom_path = None  # Not available in tuples
            else:  # Dictionary format
                study_uid = series['study_uid']
                series_uid = series['series_uid']
                dicom_path = series.get('series_path')
                sitk_image = series.get('sitk_image')
                metadata = series.get('metadata', {})

            # Create output directory
            output_dir = os.path.join(nifti_root_dir, study_uid)
            os.makedirs(output_dir, exist_ok=True)
            
            # Define output path
            output_path = os.path.join(output_dir, f"{series_uid}.nii.gz")
            
            # Skip if exists
            if not overwrite and os.path.exists(output_path):
                continue

            if dicom_path:
                convert_dicom_to_nifti(dicom_path, output_path)
                print(f"✓ Saved {output_path}")
            else:
                print(f"✗ Skipping {series_uid}: no DICOM path provided.")
            
        except Exception as e:
            print(f"✗ Failed {series_uid if 'series_uid' in locals() else 'unknown'}: {str(e)}")


def process_original_volumes(
    batch_dir: str,
    labels_csv: str,
    nifti_root_dir: str,
    pkl_path: str = "dicom_paths.pkl",
    overwrite_nifti: bool = False
) -> None:
    """Orchestrate the entire pipeline with caching."""
    # Check if NIfTI output directory is empty
    if not os.path.exists(nifti_root_dir) or not os.listdir(nifti_root_dir) or overwrite_nifti:
        print("NIfTI directory empty/overwrite requested. Processing DICOMs...")
        
        # Load or generate DICOM paths cache
        if os.path.exists(pkl_path) and not overwrite_nifti:
            with open(pkl_path, 'rb') as f:
                series_data = pickle.load(f)
            print(f"Loaded {len(series_data)} series from cache")
        else:
            series_data = save_dicom_paths(batch_dir, labels_csv, pkl_path)
            # print("Starting to load DICOM series with metadata...")
            # series_data = load_and_cache_dicom_series(batch_dir, labels_csv, pkl_path)

        
        # Convert to NIfTI
        process_series(series_data, nifti_root_dir, overwrite_nifti)
    else:
        print(f"NIfTI files already exist in {nifti_root_dir}. Skipping conversion.")

