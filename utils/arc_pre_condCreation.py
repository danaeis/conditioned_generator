import os
import pandas as pd
import SimpleITK as sitk
import numpy as np
from pathlib import Path
import shutil

def load_phase_labels(labels_path):
    """Load phase labels from CSV file."""
    df = pd.read_csv(labels_path)
    # Create a dictionary mapping series_id to phase
    phase_map = {}
    for _, row in df.iterrows():
        series_id = row['SeriesInstanceUID']
        phase = row['Label']
        phase_map[series_id] = phase
    return phase_map

def create_average_volumes(registered_dir, labels_path, output_dir):
    """Create average volumes for each phase from registered volumes."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load phase labels
    phase_map = load_phase_labels(labels_path)
    
    # Group volumes by phase
    phase_volumes = {
        'Non-contrast': [],
        'Arterial': [],
        'Venous': [],
        'Other': []
    }
    
    # List to store all volumes for overall average
    all_volumes = []
    
    # Load all registered volumes and group by phase
    for filename in os.listdir(registered_dir):
        if not filename.endswith('.nii.gz'):
            continue
            
        # Extract series_id from filename
        series_id = filename.split("_")[0]
        # series_id = filename.replace('.nii.gz', '')
        
        # Get phase for this volume
        phase = phase_map.get(series_id)
        if phase is None:
            print(f"Warning: No phase label found for {filename}")
            continue
            
        # Load volume
        volume_path = os.path.join(registered_dir, filename)
        try:
            volume = sitk.ReadImage(volume_path)
            phase_volumes[phase].append(volume)
            all_volumes.append(volume)  # Add to overall volumes list
        except Exception as e:
            print(f"Error loading {filename}: {str(e)}")
    
    # Create average volumes for each phase
    for phase, volumes in phase_volumes.items():
        if not volumes:
            print(f"No volumes found for phase: {phase}")
            continue
            
        print(f"Creating average volume for {phase} phase ({len(volumes)} volumes)")
        
        # Convert volumes to numpy arrays
        arrays = [sitk.GetArrayFromImage(vol) for vol in volumes]
        
        # Debug: Print shapes of all arrays
        print(f"\nDebugging array shapes for {phase} phase:")
        for i, arr in enumerate(arrays):
            print(f"Array {i} shape: {arr.shape}")
        
        # Compute average
        avg_array = np.mean(arrays, axis=0)
        
        # Convert back to SimpleITK image
        avg_volume = sitk.GetImageFromArray(avg_array)
        
        # Copy metadata from first volume
        avg_volume.CopyInformation(volumes[0])
        
        # Save average volume
        output_path = os.path.join(output_dir, f"average_{phase.lower()}.nii.gz")
        sitk.WriteImage(avg_volume, output_path)
        print(f"Saved average volume to {output_path}")
    
    # Create overall average volume
    if all_volumes:
        print(f"Creating overall average volume from all {len(all_volumes)} volumes")
        
        # Convert all volumes to numpy arrays
        arrays = [sitk.GetArrayFromImage(vol) for vol in all_volumes]
        
        # Compute average
        avg_array = np.mean(arrays, axis=0)
        
        # Convert back to SimpleITK image
        avg_volume = sitk.GetImageFromArray(avg_array)
        
        # Copy metadata from first volume
        avg_volume.CopyInformation(all_volumes[0])
        
        # Save average volume
        output_path = os.path.join(output_dir, "average_all.nii.gz")
        sitk.WriteImage(avg_volume, output_path)
        print(f"Saved overall average volume to {output_path}")
    else:
        print("No volumes found to create overall average")

def crop_and_save_masks(input_dir, output_dir, target_slices=512):
    """Crop each mask from the first z-slice with mask to 512 slices and save to output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    INCLUDED_LABELS = {3. , 4. , 5. , 6. , 7. }

    for filename in os.listdir(input_dir):
        if not filename.endswith('.nii.gz'):
            continue
        mask_path = os.path.join(input_dir, filename)
        try:
            mask_img = sitk.ReadImage(mask_path)
            mask_arr = sitk.GetArrayFromImage(mask_img)

            start_slice = None
            for z in reversed(range(mask_arr.shape[0])):  # assuming z-axis is the third dimension
                slice_data = mask_arr[z, :, :]
                if np.any(np.isin(slice_data, list(INCLUDED_LABELS))):
                    start_slice = z
                    break

            target_slices = 256
            print("masked arr shape", mask_arr.shape)
            cropped = mask_arr[start_slice-target_slices:start_slice, :, :]
            print("start_slice and start_slice-target_slices", start_slice, start_slice-target_slices)
            # cropped = np.transpose(cropped, (2, 1, 0))
            print("cropped shape", cropped.shape)
            # Crop from first_z to first_z+target_slices
            # cropped = mask_arr[first_z:first_z+target_slices]
            # Pad if less than target_slices
            if cropped.shape[0] < target_slices:
                pad_width = target_slices - cropped.shape[0]
                cropped = np.pad(cropped, ((0, pad_width), (0,0), (0,0)), mode='constant')
            # Convert back to image
            cropped_img = sitk.GetImageFromArray(cropped)
            # Copy spacing, origin, direction (except size)
            cropped_img.SetSpacing(mask_img.GetSpacing())
            cropped_img.SetOrigin(mask_img.GetOrigin())
            cropped_img.SetDirection(mask_img.GetDirection())
            # Save
            out_path = os.path.join(output_dir, filename)
            sitk.WriteImage(cropped_img, out_path)
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")

def create_average_volumes_from_dir(masks_dir, labels_path, output_dir):
    """Create average volumes from masks in masks_dir."""
    create_average_volumes(masks_dir, labels_path, output_dir)

if __name__ == "__main__":
    PATH = "utils/debug/ncct_cect/vindr_ds"
    # REGISTERED_DIR = os.path.join(PATH, "registered_volumes")
    REGISTERED_DIR = os.path.join(PATH, "segmentation_masks")
    LABELS_PATH = os.path.join(PATH, "labels.csv")
    OUTPUT_DIR = os.path.join(PATH, "average_volumesS")
    CROPPED_MASKS_DIR = os.path.join(PATH, "cropped_masks")

    # Step 1: Crop and save masks
    crop_and_save_masks(REGISTERED_DIR, CROPPED_MASKS_DIR, target_slices=512)
    # Step 2: Create average volumes from cropped masks
    create_average_volumes_from_dir(CROPPED_MASKS_DIR, LABELS_PATH, OUTPUT_DIR)
