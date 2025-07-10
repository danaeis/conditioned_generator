import os
import numpy as np
import SimpleITK as sitk  # pip install SimpleITK
import scipy.ndimage
import SimpleITK as sitk
import json


def shift_image_to_origin(image: sitk.Image, new_origin=(0.0, 0.0, 0.0), target_spacing=(1.0, 1.0, 1.0), is_label=False):
    original_origin = image.GetOrigin()
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    # Compute new size
    new_size = [
        max(1, int(round(osz * ospc / tspc)))
        for osz, ospc, tspc in zip(original_size, original_spacing, target_spacing)
    ]

    # Compute the shift in physical space
    shift_vector = [o - n for o, n in zip(original_origin, new_origin)]

    # Define a translation transform to shift voxels accordingly
    transform = sitk.TranslationTransform(3)
    transform.SetOffset(shift_vector)

    # Set up the resampler
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetOutputOrigin(new_origin)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetTransform(transform)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)

    shifted = resampler.Execute(image)
    return shifted


script_dir = os.path.dirname(os.path.abspath(__file__))
vol_dir = os.path.join(script_dir, "debug/ncct_cect/vindr_ds/original_volumes")
mask_dir = os.path.join(script_dir, "debug/ncct_cect/vindr_ds/total_segmentation_masks")
out_dir = os.path.join(script_dir, "debug/ncct_cect/vindr_ds/aggregated_cropped_volumes")
os.makedirs(out_dir, exist_ok=True)
print("Current working directory:", os.getcwd())

target_spacing = (1.0,1.0,1.0)
target_origin = (0.0, 0.0, 0.0)
boxes = []
required_labels_path = "utils/bundles/wholeBody_ct_segmentation/configs/required_labels.json"
with open(required_labels_path) as json_file:
    required_labels = json.load(json_file)
    required_labels = list(required_labels.values())
    print(required_labels)
    
# 1. Gather all mask files
mask_files = []
for root, _, files in os.walk(mask_dir):
    for f in files:
        if f.endswith('.nii') or f.endswith('.nii.gz'):
            mask_files.append(os.path.join(root, f))
mask_files = sorted(mask_files)


# 1. Compute global min/max for y and x axes across all masks for required labels
all_y = []
all_x = []
mask_bboxes = []  # Store per-mask z min/max for later
mask_debug_info = []
min_voxel_threshold = 100  # Set your threshold here

for mask_file in mask_files:
    mask_img_org = sitk.ReadImage(mask_file)
    mask = sitk.GetArrayFromImage(mask_img_org)
    for label in required_labels:
        binary_mask = (mask == label)
        labeled_array, num_features = scipy.ndimage.label(binary_mask)
        for comp in range(1, num_features + 1):
            comp_mask = (labeled_array == comp)
            voxel_count = np.sum(comp_mask)
            if voxel_count < min_voxel_threshold:
                # Find which z-slices this component appears in
                z_slices = np.unique(np.where(comp_mask)[0])
                print(f"Mask: {os.path.basename(mask_file)}, Label: {label}, Component: {comp}, Voxel count: {voxel_count}, Z-slices: {z_slices.tolist()}")
                mask[comp_mask] = 0

    mask_required = np.isin(mask, required_labels)
    nonzero = np.argwhere(mask_required)
    if nonzero.size == 0:
        continue
    min_coords = nonzero.min(axis=0)
    max_coords = nonzero.max(axis=0)
    # (z, y, x)
    all_y.extend([min_coords[1], max_coords[1]])
    all_x.extend([min_coords[2], max_coords[2]])
    mask_bboxes.append((mask_file, min_coords[0], max_coords[0], min_coords[1], max_coords[1], min_coords[2], max_coords[2]))
    # Debug: unique labels and counts
    unique, counts = np.unique(mask, return_counts=True)
    label_counts = dict(zip(unique.tolist(), counts.tolist()))
    mask_debug_info.append((mask_file, label_counts))

if not all_y or not all_x:
    raise RuntimeError("No non-empty masks found!")
global_ymin = min(all_y)
global_ymax = max(all_y)
global_xmin = min(all_x)
global_xmax = max(all_x)





# 2. Crop each image using per-image z min/max, but global y/x min/max
for (mask_file, zmin, zmax, ymin, ymax, xmin, xmax) in mask_bboxes:
    # Find corresponding volume file
    mask_base = os.path.basename(mask_file)
    if mask_base.endswith('_seg.nii.gz'):
        vol_base = mask_base.replace('_seg.nii.gz', '.nii.gz')
    elif mask_base.endswith('_seg.nii'):
        vol_base = mask_base.replace('_seg.nii', '.nii')
    else:
        continue
    found = False
    for case_id in os.listdir(vol_dir):
        case_folder = os.path.join(vol_dir, case_id)
        if not os.path.isdir(case_folder):
            continue
        vol_path = os.path.join(case_folder, vol_base)
        if os.path.exists(vol_path):
            found = True
            break
    if not found:
        print(f"No matching volume for mask {mask_file}")
        continue
    vol_img = sitk.ReadImage(vol_path)
    vol = sitk.GetArrayFromImage(vol_img)
    # Crop in (z, y, x) order
    cropped = vol[
        zmin:zmax+1,
        global_ymin:global_ymax+1,
        global_xmin:global_xmax+1
    ]
    # cropped = vol[
    #     zmin:zmax+1,
    #     ymin:ymax+1,
    #     xmin:xmax+1
    # ]
    # Debug: check if cropped region is empty
    if np.count_nonzero(cropped) == 0:
        print(f"WARNING: Cropped region for {vol_base} is empty!")
    cropped_img = sitk.GetImageFromArray(cropped)
    cropped_img.SetSpacing(vol_img.GetSpacing())
    cropped_img.SetDirection(vol_img.GetDirection())
    old_origin = np.array(vol_img.GetOrigin())
    old_spacing = np.array(vol_img.GetSpacing())
    old_direction = np.array(vol_img.GetDirection()).reshape(3, 3)
    index_shift = np.array([zmin, global_ymin, global_xmin])
    new_origin = old_origin + old_direction.dot(index_shift * old_spacing)
    cropped_img.SetOrigin(tuple(new_origin))
    out_path = os.path.join(out_dir, os.path.basename(vol_base))
    sitk.WriteImage(cropped_img, out_path)
    print(f"Cropped {vol_base} saved with shape: {cropped_img.GetSize()}")

    # Crop and save the segmentation mask as well
    cropped_mask = mask[
        zmin:zmax+1,
        global_ymin:global_ymax+1,
        global_xmin:global_xmax+1
    ]
    # cropped_mask = mask[
    #     zmin:zmax+1,
    #     ymin:ymax+1,
    #     xmin:xmax+1
    # ]
    # Only keep required labels
    cropped_mask[~np.isin(cropped_mask, required_labels)] = 0
    cropped_mask_img = sitk.GetImageFromArray(cropped_mask)
    cropped_mask_img.CopyInformation(cropped_img)
    mask_out_path = os.path.join(out_dir, os.path.basename(mask_base))
    sitk.WriteImage(cropped_mask_img, mask_out_path)
    print(f"Cropped mask {mask_base} saved with shape: {cropped_mask_img.GetSize()}")


# # 3. Debug output for unique mask labels and their counts
# print("\n--- Mask label counts per file ---")
# for mask_file, label_counts in mask_debug_info:
#     print(f"{os.path.basename(mask_file)}: {label_counts}")

