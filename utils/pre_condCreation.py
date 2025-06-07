import os
import pandas as pd
import SimpleITK as sitk
import numpy as np
from pathlib import Path

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
        if not filename.endswith('_registered.nii.gz'):
            continue
            
        # Extract series_id from filename
        series_id = filename.replace('_registered.nii.gz', '')
        
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

if __name__ == "__main__":
    PATH = "utils/debug/ncct_cect/vindr_ds"
    REGISTERED_DIR = os.path.join(PATH, "registered_volumes")
    LABELS_PATH = os.path.join(PATH, "labels.csv")
    OUTPUT_DIR = os.path.join(PATH, "average_volumes")
    
    create_average_volumes(REGISTERED_DIR, LABELS_PATH, OUTPUT_DIR)
