import os
import logging
import SimpleITK as sitk
from registration_utils import compute_quick_metric, register_to_atlas  # assuming the util file is named utils.py

def load_images_from_directory(input_dir):
    """
    Load all NIfTI volumes from the nested input directory: study_id/series_id.nii
    """
    volume_paths = {}
    for study_id in os.listdir(input_dir):
        study_path = os.path.join(input_dir, study_id)
        if not os.path.isdir(study_path):
            continue
        for file in os.listdir(study_path):
            if file.endswith(".nii") or file.endswith(".nii.gz"):
                full_path = os.path.join(study_path, file)
                series_id = os.path.splitext(file)[0].replace(".nii", "").replace(".gz", "")
                volume_paths[(study_id, series_id)] = full_path
    return volume_paths

def select_atlas(volumes):
    """
    Select the most representative volume (atlas) using average similarity.
    """
    logging.info("Selecting atlas from candidate volumes...")
    volume_keys = list(volumes.keys())
    metrics = {}
    for i in range(len(volume_keys)):
        ref_key = volume_keys[i]
        ref_img = volumes[ref_key]
        total_metric = 0
        for j in range(len(volume_keys)):
            if i == j:
                continue
            cmp_img = volumes[volume_keys[j]]
            metric = compute_quick_metric(ref_img, cmp_img, metric_type='ncc')
            total_metric += metric
        metrics[ref_key] = total_metric
    
    best_key = min(metrics, key=metrics.get)
    logging.info(f"Selected atlas: {best_key}")
    return best_key, volumes[best_key]

def main(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    volume_paths = load_images_from_directory(input_dir)
    
    # Load all images into memory
    volumes = {
        key: sitk.ReadImage(path)
        for key, path in volume_paths.items()
    }
    
    # Select atlas
    atlas_key, atlas_img = select_atlas(volumes)

    for key, moving_img in volumes.items():
        if key == atlas_key:
            # Copy atlas itself to output unchanged
            sitk.WriteImage(atlas_img, os.path.join(output_dir, f"{key[0]}_{key[1]}_registered.nii.gz"))
            continue
        
        logging.info(f"Registering {key} to atlas {atlas_key}...")
        transform, registered_img = register_to_atlas(atlas_img, moving_img)
        
        out_path = os.path.join(output_dir, f"{key[0]}_{key[1]}_registered.nii.gz")
        sitk.WriteImage(registered_img, out_path)
        logging.info(f"Saved registered image to {out_path}")

if __name__ == "__main__":
    # import argparse
    # parser = argparse.ArgumentParser(description="Register CT volumes to an automatically selected atlas.")
    # parser.add_argument('--input_dir', type=str, required=True, help='Input directory with study/series NIfTI files.')
    # parser.add_argument('--output_dir', type=str, required=True, help='Where to save registered images.')

    # args = parser.parse_args()
    # main(args.input_dir, args.output_dir)
    INPUT_DIR, OUTPUT_DIR = "debug/ncct_cect/vindr_ds/original_volumes", "debug/ncct_cect/vindr_ds/registered_volumes"
    main(INPUT_DIR, OUTPUT_DIR)