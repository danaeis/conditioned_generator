import SimpleITK as sitk
import numpy as np
import os

def crop_center(image: sitk.Image, crop_size):
    """
    Crop center region of size crop_size (x, y, z) from the image.
    """
    size = image.GetSize()
    start = [(size[i] - crop_size[i]) // 2 for i in range(3)]
    end = [start[i] + crop_size[i] for i in range(3)]

    region_extractor = sitk.RegionOfInterestImageFilter()
    region_extractor.SetSize(crop_size)
    region_extractor.SetIndex(start)

    return region_extractor.Execute(image)

def resample_spacing(image, new_spacing=(1.0, 1.0, 1.0)):
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    new_size = [int(round(osz * ospc / nspc)) for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)]
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetDefaultPixelValue(0)
    
    return resampler.Execute(image)

def detect_abdomen_bbox(image, lower=-200, upper=200, margin=10):
    array = sitk.GetArrayFromImage(image)
    mask = ((array > lower) & (array < upper)).astype(np.uint8)

    coords = np.argwhere(mask)
    if coords.size == 0:
        return image  # fallback to full image

    min_z, min_y, min_x = coords.min(axis=0)
    max_z, max_y, max_x = coords.max(axis=0)

    start = [int(max(min_x - margin, 0)), int(max(min_y - margin, 0)), int(max(min_z - margin, 0))]
    end = [int(min(max_x + margin, image.GetSize()[0])),
           int(min(max_y + margin, image.GetSize()[1])),
           int(min(max_z + margin, image.GetSize()[2]))]
    
    size = [e - s for s, e in zip(start, end)]

    extractor = sitk.RegionOfInterestImageFilter()
    extractor.SetIndex(start)
    extractor.SetSize(size)
    
    return extractor.Execute(image)


def resample_to_target(image: sitk.Image, target: sitk.Image):
    """
    Resample image to match target's spacing, size, and orientation.
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(target)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)
    return resampler.Execute(image)

def resample_to_fixed_shape(image, target_shape=(128, 128, 128)):
    spacing = image.GetSpacing()
    size = target_shape
    physical_size = [sz * sp for sz, sp in zip(size, spacing)]
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(size)
    resampler.SetOutputSpacing(spacing)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    
    return resampler.Execute(image)

def compute_average_volume(image_paths):
    """
    Load and compute average of all images at image_paths.
    Returns a SimpleITK.Image of the average.
    """
    arrays = []
    for path in image_paths:
        img = sitk.ReadImage(path)
        arr = sitk.GetArrayFromImage(img).astype(np.float32)
        arrays.append(arr)

    average_array = np.mean(np.stack(arrays), axis=0)
    avg_img = sitk.GetImageFromArray(average_array)
    # Set same metadata from one of the inputs
    ref_img = sitk.ReadImage(image_paths[0])
    avg_img.CopyInformation(ref_img)
    return avg_img
