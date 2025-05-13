import os
import pickle
import pandas as pd
import SimpleITK as sitk
from dicom_utils import load_dicom_series

def load_and_cache_dicom_series(batch_dir, labels_csv, cache_path="dicom_series.pkl"):
    if os.path.exists(cache_path):
        print(f"Loading cached DICOM series from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print("Scanning DICOM directories and loading volumes...")
    labels_df = pd.read_csv(labels_csv).dropna(subset=['StudyInstanceUID', 'SeriesInstanceUID'])
    label_dict = {
        (row['StudyInstanceUID'], row['SeriesInstanceUID']): row['Label']
        for _, row in labels_df.iterrows()
    }

    dicom_series = []
    z_spacings = []

    for batch in os.listdir(batch_dir):
        batch_path = os.path.join(batch_dir, batch)
        for study in os.listdir(batch_path):
            study_path = os.path.join(batch_path, study)
            for series in os.listdir(study_path):
                series_path = os.path.join(study_path, series)
                if (study, series) in label_dict:
                    try:
                        image_3d, spacing, origin, direction = load_dicom_series(series_path)
                        image = sitk.GetImageFromArray(image_3d)
                        image.SetSpacing(spacing)
                        image.SetOrigin(origin)
                        image.SetDirection(direction)
                        dicom_series.append((study, series, image))
                        z_spacings.append(spacing[2])
                    except Exception as e:
                        print(f"Failed to load {study}/{series}: {e}")

    with open(cache_path, "wb") as f:
        pickle.dump((dicom_series, z_spacings), f)

    print(f"Found {len(dicom_series)} matching DICOM series.")
    print(f"Saved to cache: {cache_path}")
    return dicom_series, z_spacings
