import os
import pickle
import pandas as pd
import SimpleITK as sitk
from dicom_utils import load_dicom_series

def load_and_cache_dicom_series(batch_dir, labels_csv, cache_path="dicom_series.pkl"):
    if os.path.exists(cache_path):
        print(f"Loading cached DICOM series from {cache_path}")
        try:
            with open(cache_path, "rb") as f:
                dicom_series_data = pickle.load(f)
                # Validate the loaded data format
                # Expecting a list of tuples: (study_uid, series_uid, sitk_volume, metadata)
                is_valid_cache_format = (
                    isinstance(dicom_series_data, list) and
                    dicom_series_data and # Check if list is not empty
                    isinstance(dicom_series_data[0], tuple) and
                    len(dicom_series_data[0]) == 4
                )

                if not is_valid_cache_format:
                     raise ValueError("Cached data format is unexpected or empty. Rebuilding cache.")
                     
                dicom_series_list = dicom_series_data
                print("Successfully loaded cached data with metadata.")
                # No need to return z_spacings separately with the new structure
                # z_spacings = [img.GetSpacing()[2] for _, _, img, _ in dicom_series_list]
                return dicom_series_list
                
        except (EOFError, ValueError, pickle.UnpicklingError, IndexError) as e:
             # Catch IndexError here as well to handle cases of empty or malformed lists
             print(f"Error loading cached data ({e}). Rebuilding cache.")
             # Proceed to rebuild cache if loading failed or format was wrong
             pass
    
    print("Scanning DICOM directories and loading volumes with metadata...")

    # Step 1: load label info
    labels_df = pd.read_csv(labels_csv).dropna(subset=['StudyInstanceUID', 'SeriesInstanceUID', 'Label'])
    phase_lookup = {
        (row['StudyInstanceUID'], row['SeriesInstanceUID']): row['Label'].lower()
        for _, row in labels_df.iterrows()
    }

    # Step 2: Build (StudyUID, Phase) → best SeriesUID based on Z-spacing and collect data
    best_series_dict = {}  # (StudyUID, Phase) → (spacing, series_uid, sitk_image, metadata)
    all_series_data = [] # List to store (study_uid, series_uid, series_path) to load individually

    for batch in os.listdir(batch_dir):
        batch_path = os.path.join(batch_dir, batch)
        for study in os.listdir(batch_path):
            study_path = os.path.join(batch_path, study)
            for series in os.listdir(study_path):
                series_path = os.path.join(study_path, series)

                key = (study, series)
                if key not in phase_lookup:
                    continue  # this series is not in the CSV

                phase = phase_lookup[key]

                # Collect all series paths first, then process to get spacing and select best
                all_series_data.append((study, series, series_path, phase))

    # Now iterate through collected series data to load and select the best
    processed_series_data = [] # To store (study_uid, series_uid, sitk_image, metadata)
    for study, series, series_path, phase in all_series_data:
        try:
            # Use the updated load_dicom_series which returns image and metadata
            image, metadata = load_dicom_series(series_path)
            z_spacing = image.GetSpacing()[2]

            current_best = best_series_dict.get((study, phase), (float("inf"), None, None, None))
            if z_spacing < current_best[0]:  # prefer finer slices
                best_series_dict[(study, phase)] = (z_spacing, series, image, metadata)

        except Exception as e:
            print(f"Failed to load {study}/{series} from {series_path}: {e}")

    # Step 3: Collect filtered results including metadata
    dicom_series_list = []
    for (study, phase), (spacing, series, image, metadata) in best_series_dict.items():
        dicom_series_list.append((study, series, image, metadata))

    # Step 4: Cache results (new format including metadata)
    if dicom_series_list:
        print(f"Caching {len(dicom_series_list)} selected series with metadata to {cache_path}")
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(dicom_series_list, f)
            print(f"Saved to cache: {cache_path}")
        except Exception as e:
            print(f"Error saving cache file {cache_path}: {e}")
    else:
        print("No DICOM series were successfully loaded to cache.")

    print(f"Selected {len(dicom_series_list)} best series (1 per Study x Phase).")

    # Step 5: Compute average number of slices (optional, for logging)
    slice_counts = [image.GetSize()[2] for _, _, image, _ in dicom_series_list]
    avg_slices = sum(slice_counts) / len(slice_counts) if slice_counts else 0
    print(f"Average number of slices: {avg_slices:.2f}")

    return dicom_series_list
