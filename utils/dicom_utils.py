import os
import pickle
import pydicom
import pandas as pd
import SimpleITK as sitk
import nibabel as nib
import numpy as np
from typing import List, Dict, Optional
from collections import defaultdict
import json
import subprocess


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



def convert_dicom_to_nifti_dcm2niix(
    dicom_path: str,
    output_path: str,
    overwrite: bool = False
) -> None:
    """
    Convert DICOM series to NIfTI using dcm2niix.
    
    Args:
        dicom_path (str): Path to the DICOM directory
        output_path (str): Path where the NIfTI file should be saved
        overwrite (bool): Whether to overwrite existing files
    """
    try:
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        
        # Get base filename without any extensions
        base_filename = os.path.splitext(os.path.splitext(os.path.basename(output_path))[0])[0]
        
        # Prepare dcm2niix command - simplified for compatibility
        cmd = [
            "dcm2niix",
            "-z", "y",
            "-o", output_dir,  # output directory
            "-f", base_filename,  # output filename without extension
            dicom_path  # input DICOM folder
        ]
        
        # Debug: Print the full command
        print(f"Executing command: {' '.join(cmd)}")
        
        # Debug: Check if input directory exists and has DICOM files
        if not os.path.exists(dicom_path):
            raise Exception(f"Input directory does not exist: {dicom_path}")
        
        dicom_files = [f for f in os.listdir(dicom_path) if f.endswith('.dcm')]
        if not dicom_files:
            raise Exception(f"No DICOM files found in {dicom_path}")
        print(f"Found {len(dicom_files)} DICOM files in input directory")
        
        # Run dcm2niix
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Debug: Print command output
        print(f"Command stdout: {result.stdout}")
        print(f"Command stderr: {result.stderr}")
        
        if result.returncode != 0:
            raise Exception(f"dcm2niix failed: {result.stderr}")
        
        # Debug: Check if output file was created
        expected_output = os.path.join(output_dir, f"{base_filename}.nii.gz")
        if not os.path.exists(expected_output):
            raise Exception(f"Expected output file not created: {expected_output}")
            
        print(f"✓ Successfully converted {dicom_path} to {output_path}")
        
    except Exception as e:
        print(f"✗ Failed to convert {dicom_path}: {str(e)}")
        raise

def process_series(
    series_data: List,  # Accepts both tuples and dicts
    nifti_root_dir: str,
    overwrite: bool = False,
    use_dcm2niix: bool = True  # New parameter to choose conversion method
) -> None:
    """
    Process DICOM series in either tuple or dictionary format.
    Tuple format: (study_uid, series_uid, sitk_image, metadata)
    Dict format: {'study_uid': ..., 'series_uid': ..., ...}
    
    Args:
        series_data (List): List of series data in either tuple or dict format
        nifti_root_dir (str): Root directory for NIfTI output
        overwrite (bool): Whether to overwrite existing files
        use_dcm2niix (bool): Whether to use dcm2niix for conversion (True) or SimpleITK (False)
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
            
            # Skip if exists and not overwriting
            if not overwrite and os.path.exists(output_path):
                print(f"✓ Skipping {output_path} (already exists)")
                continue

            if dicom_path:
                convert_dicom_to_nifti_dcm2niix(dicom_path, output_path, overwrite)
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

