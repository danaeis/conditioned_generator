import os
import numpy as np
import SimpleITK as sitk  # pip install SimpleITK
import scipy.ndimage
import SimpleITK as sitk


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
mask_dir = os.path.join(script_dir, "debug/ncct_cect/vindr_ds/segmentation_masks")
out_dir = os.path.join(script_dir, "debug/ncct_cect/vindr_ds/aggregated_cropped_volumes")
os.makedirs(out_dir, exist_ok=True)
print("Current working directory:", os.getcwd())

target_spacing = (1.0,1.0,1.0)
target_origin = (0.0, 0.0, 0.0)
# 1. Gather all mask files
mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.nii') or f.endswith('.nii.gz')])

boxes = []

# 2. Compute bounding boxes for each mask
for mask_file in mask_files:
    mask_path = os.path.join(mask_dir, mask_file)
    mask_img_org = sitk.ReadImage(mask_path)
    print(f"mask volume {mask_file} size: {mask_img_org.GetSize()}")
    print(f"mask volume {mask_file} spacing: {mask_img_org.GetSpacing()}")
    print(f"mask volume {mask_file} origin: {mask_img_org.GetOrigin()}")
    print(f"mask volume {mask_file} direction: {mask_img_org.GetDirection()}")
    mask_img = shift_image_to_origin(mask_img_org, target_origin, target_spacing, True)
    print(f"mask volume {mask_file} after shift size: {mask_img.GetSize()}")
    print(f"mask volume {mask_file} after shift spacing: {mask_img.GetSpacing()}")
    print(f"mask volume {mask_file} after shift origin: {mask_img.GetOrigin()}")
    print(f"mask volume {mask_file} after shift direction: {mask_img.GetDirection()}")
    mask = sitk.GetArrayFromImage(mask_img_org)
    # mask = np.transpose(mask, (2, 1, 0))  # sitk is z,y,x; numpy is x,y,z
    # Remove small connected components
    labeled_mask, num_features = scipy.ndimage.label(mask > 2)
    component_sizes = np.bincount(labeled_mask.ravel())
    # print(component_sizes)
    # Zero is background, so skip it
    too_small = component_sizes < 500
    # too_small[0] = False
    mask_cleaned = mask.copy()
    mask_cleaned[too_small[labeled_mask]] = 0
    mask = mask_cleaned
    # if np.count_nonzero(mask) < 500:  # adjust this threshold as needed
    #     print(f"Skipping {mask_file} due to too few foreground pixels.")
    #     continue
    
    nonzero = np.argwhere(mask > 2)
    if nonzero.size == 0:
        continue  # skip empty masks
    min_coords = nonzero.min(axis=0)
    max_coords = nonzero.max(axis=0)
    boxes.append(np.concatenate([min_coords, max_coords]))

    # Save the bounding box as a mask
    bbox_mask = np.zeros_like(mask, dtype=np.uint8)
    bbox_mask[
        min_coords[0]:max_coords[0]+1,
        min_coords[1]:max_coords[1]+1,
        min_coords[2]:max_coords[2]+1
    ] = 1
    bbox_mask_img = sitk.GetImageFromArray(bbox_mask)
    bbox_mask_img.CopyInformation(mask_img_org)
    mask_file_name = mask_file.replace(".nii.gz", "")
    out_path = os.path.join(out_dir, f"{mask_file_name}_bbox_mask.nii.gz")
    sitk.WriteImage(bbox_mask_img, out_path)

boxes = np.array(boxes)
if len(boxes) == 0:
    raise RuntimeError("No non-empty masks found!")

# 3. Aggregate bounding boxes using percentiles
loose_min = np.percentile(boxes[:, :3], 5, axis=0).astype(int)
loose_max = np.percentile(boxes[:, 3:], 95, axis=0).astype(int)

# 4. Loop over all subfolders in original_volumes
for case_id in sorted(os.listdir(vol_dir)):
    case_folder = os.path.join(vol_dir, case_id)
    if not os.path.isdir(case_folder):
        continue
    for vol_file in os.listdir(case_folder):
        if not (vol_file.endswith('.nii') or vol_file.endswith('.nii.gz')):
            continue
        vol_path = os.path.join(case_folder, vol_file)
        # Construct mask file name based on case_id and extension
        ext = '.nii.gz' if vol_file.endswith('.nii.gz') else '.nii'
        vol_file_base = vol_file.replace(ext,"")
        mask_file = vol_file_base + '_seg' + ext
        mask_path = os.path.join(mask_dir, mask_file)
        if not os.path.exists(mask_path):
            continue
        mask_img = sitk.ReadImage(mask_path)
        mask = sitk.GetArrayFromImage(mask_img)
        # mask = np.transpose(mask, (2, 1, 0))
        if np.count_nonzero(mask) < 500:  # adjust this threshold as needed
            print(f"Skipping {mask_file} due to too few foreground pixels.")
            continue
        vol_img = sitk.ReadImage(vol_path)
        print(f"volume {vol_file_base} size: {vol_img.GetSize()}")
        print(f"volume {vol_file_base} spacing: {vol_img.GetSpacing()}")
        print(f"volume {vol_file_base} origin: {vol_img.GetOrigin()}")
        print(f"volume {vol_file_base} direction: {vol_img.GetDirection()}")
        vol_img = shift_image_to_origin(vol_img, target_origin, target_spacing)
        vol = sitk.GetArrayFromImage(vol_img)
        # vol = np.transpose(vol, (2, 1, 0))
        print(f"volume {vol_file_base}after shifting size: {vol_img.GetSize()}")
        print(f"volume {vol_file_base} after shifting spacing: {vol_img.GetSpacing()}")
        print(f"volume {vol_file_base} after shifting origin: {vol_img.GetOrigin()}")
        print(f"volume {vol_file_base} after shifting direction: {vol_img.GetDirection()}")
        cropped = vol[
            loose_min[0]:loose_max[0]+1,
            loose_min[1]:loose_max[1]+1,
            loose_min[2]:loose_max[2]+1
        ]
        # Save cropped volume
        # cropped = np.transpose(cropped, (2, 1, 0))  # back to z,y,x for sitk
        cropped_img = sitk.GetImageFromArray(cropped)
        # After cropping and before saving:
        
        cropped_img = sitk.GetImageFromArray(cropped)
        cropped_img.SetSpacing(vol_img.GetSpacing())
        cropped_img.SetDirection(vol_img.GetDirection())

        # Calculate new origin
        old_origin = np.array(vol_img.GetOrigin())
        old_spacing = np.array(vol_img.GetSpacing())
        old_direction = np.array(vol_img.GetDirection()).reshape(3, 3)
        # loose_min is in (x, y, z) index order
        index_shift = np.array([loose_min[0], loose_min[1], loose_min[2]])
        new_origin = old_origin + old_direction.dot(index_shift * old_spacing)
        cropped_img.SetOrigin(tuple(new_origin))

        out_path = os.path.join(out_dir, vol_file_base+ext)
        sitk.WriteImage(cropped_img, out_path)
        print(f"Cropped volume {vol_file_base} saved \n with shape:{cropped_img.GetSize()}")
    print("boxes", boxes)
print("All done!")

cropped_dir = out_dir
padded_dir =  os.path.join(script_dir, 'debug/ncct_cect/vindr_ds/padded_volumes')
os.makedirs(padded_dir, exist_ok=True)

# 1. Find the target shape
shapes = []
for fname in os.listdir(cropped_dir):
    if fname.endswith('.nii') or fname.endswith('.nii.gz'):
        img = sitk.ReadImage(os.path.join(cropped_dir, fname))
        vol = sitk.GetArrayFromImage(img)
        # vol = np.transpose(vol, (2, 1, 0))
        shapes.append(vol.shape)
target_shape = np.max(np.array(shapes), axis=0)

for fname in os.listdir(cropped_dir):
    if fname.endswith('.nii') or fname.endswith('.nii.gz'):
        img = sitk.ReadImage(os.path.join(cropped_dir, fname))
        vol = sitk.GetArrayFromImage(img)
        # vol = np.transpose(vol, (2, 1, 0))
        pad_width = [
            (0, target_shape[0] - vol.shape[0]),
            (0, target_shape[1] - vol.shape[1]),
            (0, target_shape[2] - vol.shape[2])
        ]
        vol_padded = np.pad(vol, pad_width, mode='constant', constant_values=0)
        # vol_padded = np.transpose(vol_padded, (2, 1, 0))  # back to z,y,x for sitk
        padded_img = sitk.GetImageFromArray(vol_padded)
        
        padded_img.SetOrigin(img.GetOrigin())
        padded_img.SetSpacing(img.GetSpacing())
        padded_img.SetDirection(img.GetDirection())

        sitk.WriteImage(padded_img, os.path.join(padded_dir, fname))
        print(f"Padded {fname} to shape {padded_img.GetSize()}")
