import os
import pickle
import pandas as pd
import SimpleITK as sitk
from dicom_utils import load_dicom_series

def load_and_cache_dicom_series(batch_dir, labels_csv, cache_path="dicom_series.pkl"):
    if os.path.exists(cache_path):
        print(f"Loading cached DICOM series from {cache_path}")
        with open(cache_path, "rb") as f:
            dicom_series, z_spacings = pickle.load(f)
    else:
        print("Scanning DICOM directories and loading volumes...")

        # Step 1: load label info
        labels_df = pd.read_csv(labels_csv).dropna(subset=['StudyInstanceUID', 'SeriesInstanceUID', 'Label'])
        phase_lookup = {
            (row['StudyInstanceUID'], row['SeriesInstanceUID']): row['Label'].lower()
            for _, row in labels_df.iterrows()
        }

        # Step 2: Build (StudyUID, Phase) → best SeriesUID based on Z-spacing
        best_series_dict = {}  # (StudyUID, Phase) → (spacing, series, image)
        dicom_series = []
        z_spacings = []

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
                    try:
                        image_3d, spacing, origin, direction = load_dicom_series(series_path)
                        z_spacing = spacing[2]

                        current_best = best_series_dict.get((study, phase), (float("inf"), None, None))
                        if z_spacing < current_best[0]:  # prefer finer slices
                            image = sitk.GetImageFromArray(image_3d)
                            image.SetSpacing(spacing)
                            image.SetOrigin(origin)
                            image.SetDirection(direction)
                            best_series_dict[(study, phase)] = (z_spacing, series, image)

                    except Exception as e:
                        print(f"Failed to load {study}/{series}: {e}")

        # Step 3: Collect filtered results
        for (study, phase), (spacing, series, image) in best_series_dict.items():
            dicom_series.append((study, series, image))
            z_spacings.append(spacing)

        # Step 4: Cache results
        with open(cache_path, "wb") as f:
            pickle.dump((dicom_series, z_spacings), f)

        print(f"Selected {len(dicom_series)} best series (1 per Study x Phase).")
        print(f"Saved to cache: {cache_path}")

    # Step 5: Compute average number of slices
    slice_counts = [image.GetSize()[2] for _, _, image in dicom_series]
    avg_slices = sum(slice_counts) / len(slice_counts) if slice_counts else 0
    print(f"Average number of slices: {avg_slices:.2f}")

    return dicom_series, z_spacings
