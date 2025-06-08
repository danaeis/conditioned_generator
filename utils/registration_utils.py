import SimpleITK as sitk
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import logging
import time
import os
from tqdm import tqdm
from datetime import datetime

# Set up logging with a cleaner format
log_dir = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'registration_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file)
    ]
)

# Create a custom logger for registration progress
reg_logger = logging.getLogger('registration')
reg_logger.setLevel(logging.INFO)

def log_progress(message, level='info'):
    """Helper function for consistent logging with timestamps"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    if level == 'info':
        reg_logger.info(f"[{timestamp}] {message}")
    elif level == 'warning':
        reg_logger.warning(f"[{timestamp}] {message}")
    elif level == 'error':
        reg_logger.error(f"[{timestamp}] {message}")

# Check for GPU support and set threads
try:
    # Try to set number of threads
    if hasattr(sitk, 'ProcessObject_SetGlobalDefaultNumberOfThreads'):
        sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(4)
        logging.info("Set SimpleITK to use 4 threads")
    else:
        logging.warning("SimpleITK version does not support thread control")
except Exception as e:
    logging.warning(f"Could not set number of threads: {str(e)}")

def normalize_intensity(image, min_percentile=1, max_percentile=99):
    """
    Normalize image intensity to [0,1] range using percentile-based clipping.
    
    Args:
        image: SimpleITK image
        min_percentile: Lower percentile for clipping (default: 1)
        max_percentile: Upper percentile for clipping (default: 99)
    
    Returns:
        Normalized SimpleITK image
    """
    # Convert to numpy array for percentile calculation
    img_array = sitk.GetArrayFromImage(image)
    
    # Calculate percentiles
    min_val = np.percentile(img_array, min_percentile)
    max_val = np.percentile(img_array, max_percentile)
    
    # Clip values
    img_array = np.clip(img_array, min_val, max_val)
    
    # Normalize to [0,1]
    img_array = (img_array - min_val) / (max_val - min_val)
    
    # Ensure values are strictly in [0,1] range
    img_array = np.clip(img_array, 0.0, 1.0)
    
    # Convert back to SimpleITK image
    normalized_image = sitk.GetImageFromArray(img_array)
    normalized_image.CopyInformation(image)
    
    # Log normalization statistics
    log_progress(f"Normalization stats - min: {np.min(img_array):.3f}, max: {np.max(img_array):.3f}, mean: {np.mean(img_array):.3f}")
    
    return normalized_image

def add_padding(image, padding_size=10):
    """
    Add padding to the image to prevent edge artifacts during registration.
    
    Args:
        image: SimpleITK image
        padding_size: Number of voxels to pad in each dimension (default: 10)
    
    Returns:
        Padded SimpleITK image
    """
    # Get image size
    size = image.GetSize()
    
    # Calculate new size with padding
    new_size = [s + 2 * padding_size for s in size]
    
    # Create padded image
    padded_image = sitk.ConstantPad(image, 
                                  [padding_size] * len(size),
                                  [padding_size] * len(size),
                                  image.GetPixelIDValue())
    
    return padded_image

def preprocess_image(image, normalize=True, pad=True, padding_size=10):
    """
    Apply preprocessing steps to the image.
    
    Args:
        image: SimpleITK image
        normalize: Whether to normalize intensity (default: True)
        pad: Whether to add padding (default: True)
        padding_size: Size of padding if padding is enabled (default: 10)
    
    Returns:
        Preprocessed SimpleITK image
    """
    processed_image = image
    
    if normalize:
        logging.info("Normalizing image intensity...")
        processed_image = normalize_intensity(processed_image)
    
    if pad:
        logging.info(f"Adding padding of size {padding_size}...")
        processed_image = add_padding(processed_image, padding_size)
    
    return processed_image

def compute_quick_metric(fixed_image, moving_image, metric_type='ncc'):
    """
    Compute a quick similarity metric between images for atlas selection.
    Uses downsampled images for speed.
    """
    start_time = time.time()
    
    # Cast images to float32 for registration compatibility
    fixed_image = sitk.Cast(fixed_image, sitk.sitkFloat32)
    moving_image = sitk.Cast(moving_image, sitk.sitkFloat32)
    
    # Downsample images for quick comparison
    shrink_factors = [8, 8, 8]  # Increased downsampling for speed
    fixed_down = sitk.Shrink(fixed_image, shrink_factors)
    moving_down = sitk.Shrink(moving_image, shrink_factors)
    
    if metric_type == 'ncc':
        # Normalized Cross Correlation using ImageRegistrationMethod
        registration_method = sitk.ImageRegistrationMethod()
        registration_method.SetMetricAsCorrelation()
        registration_method.SetInterpolator(sitk.sitkLinear)
        registration_method.SetOptimizerAsGradientDescent(
            learningRate=1.0,
            numberOfIterations=20,  # Reduced iterations
            convergenceMinimumValue=1e-4,  # Relaxed convergence
            convergenceWindowSize=5
        )
        
        # Use identity transform for quick metric computation
        transform = sitk.CenteredTransformInitializer(
            fixed_down, moving_down, sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )
        registration_method.SetInitialTransform(transform)
        
        # Get metric value without full registration
        metric_value = registration_method.MetricEvaluate(fixed_down, moving_down)
        elapsed_time = time.time() - start_time
        logging.info(f"Metric computation took {elapsed_time:.2f} seconds")
        return -metric_value
    else:
        # Mutual Information (faster version)
        registration_method = sitk.ImageRegistrationMethod()
        registration_method.SetMetricAsMattesMutualInformation(16)  # Reduced bins
        registration_method.SetInterpolator(sitk.sitkLinear)
        registration_method.SetOptimizerAsGradientDescent(
            learningRate=1.0,
            numberOfIterations=20,  # Reduced iterations
            convergenceMinimumValue=1e-4,  # Relaxed convergence
            convergenceWindowSize=5
        )
        
        # Use identity transform for quick metric computation
        transform = sitk.CenteredTransformInitializer(
            fixed_down, moving_down, sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )
        registration_method.SetInitialTransform(transform)
        
        # Get metric value without full registration
        metric_value = registration_method.MetricEvaluate(fixed_down, moving_down)
        elapsed_time = time.time() - start_time
        logging.info(f"Metric computation took {elapsed_time:.2f} seconds")
        return -metric_value

def register_to_atlas(fixed_image, moving_image, transform_type='multi_step', is_atlas=False):
    """
    Enhanced registration with multi-step approach and optimized parameters.
    
    Args:
        fixed_image: The atlas image
        moving_image: The image to register
        transform_type: Type of registration to perform
        is_atlas: Whether the moving image is the atlas itself
    """
    start_time = time.time()
    
    # Cast images to float32 for registration compatibility
    log_progress("Starting registration process...")
    fixed_image = sitk.Cast(fixed_image, sitk.sitkFloat32)
    moving_image = sitk.Cast(moving_image, sitk.sitkFloat32)
    
    # Store original fixed image size for final resampling
    original_size = fixed_image.GetSize()
    original_spacing = fixed_image.GetSpacing()
    original_direction = fixed_image.GetDirection()
    original_origin = fixed_image.GetOrigin()
    
    log_progress(f"Atlas dimensions: {original_size}")
    log_progress(f"Moving image dimensions: {moving_image.GetSize()}")
    
    # Apply preprocessing to both images
    log_progress("Preprocessing images...")
    fixed_image = preprocess_image(fixed_image, normalize=True, pad=True)
    moving_image = preprocess_image(moving_image, normalize=True, pad=True)
    
    if is_atlas:
        # For atlas, just resample to ensure consistent dimensions
        log_progress("Processing atlas image...")
        registered_image = sitk.Resample(
            moving_image,
            original_size,
            sitk.Transform(3, sitk.sitkIdentity),
            sitk.sitkBSpline,
            original_origin,
            original_spacing,
            original_direction,
            0.0,
            moving_image.GetPixelID()
        )
        log_progress("✓ Atlas preprocessing completed")
        return sitk.Transform(3, sitk.sitkIdentity), registered_image
    
    if transform_type == 'multi_step':
        # Step 1: Rigid registration
        log_progress("Step 1/3: Performing rigid registration...")
        rigid_transform = perform_rigid_registration(fixed_image, moving_image)
        log_progress("✓ Rigid registration completed")
        
        # Step 2: Affine registration using rigid result as initial transform
        log_progress("Step 2/3: Performing affine registration...")
        affine_transform = perform_affine_registration(fixed_image, moving_image, rigid_transform)
        log_progress("✓ Affine registration completed")
        
        # Step 3: BSpline registration (optional, more accurate but slower)
        log_progress("Step 3/3: Performing BSpline registration...")
        final_transform = perform_bspline_registration(fixed_image, moving_image, affine_transform)
        log_progress("✓ BSpline registration completed")
        # final_transform = affine_transform
        
        # Resample using final transform with explicit size and spacing
        log_progress("Resampling final image...")
        log_progress(f"Resampling to size: {original_size}")
        log_progress(f"Resampling to spacing: {original_spacing}")
        
        registered_image = sitk.Resample(
            moving_image,
            original_size,  # Use original fixed image size
            final_transform,
            sitk.sitkBSpline,
            original_origin,
            original_spacing,
            original_direction,
            0.0,
            moving_image.GetPixelID()
        )
        
        # Verify dimensions
        if registered_image.GetSize() != original_size:
            log_progress(f"Warning: Resampled image size {registered_image.GetSize()} does not match atlas size {original_size}", level='warning')
        
        elapsed_time = time.time() - start_time
        log_progress(f"✓ Registration completed in {elapsed_time:.2f} seconds")
        return final_transform, registered_image
    else:
        # Fallback to original registration method
        log_progress("Using fallback registration method...")
        return perform_affine_registration(fixed_image, moving_image)

class RegistrationProgressCallback:
    def __init__(self, total_iterations):
        self.pbar = tqdm(total=total_iterations, desc="Optimizing transform", leave=False)
        self.current_iteration = 0
        
    def __call__(self):
        self.current_iteration += 1
        self.pbar.update(1)
        
    def close(self):
        self.pbar.close()

def perform_rigid_registration(fixed_image, moving_image, initial_transform=None):
    """Perform rigid registration with optimized parameters."""
    start_time = time.time()
    
    # Cast images to float32 for registration compatibility
    fixed_image = sitk.Cast(fixed_image, sitk.sitkFloat32)
    moving_image = sitk.Cast(moving_image, sitk.sitkFloat32)
    
    # Log image information
    log_progress(f"Fixed image size: {fixed_image.GetSize()}, spacing: {fixed_image.GetSpacing()}")
    log_progress(f"Moving image size: {moving_image.GetSize()}, spacing: {moving_image.GetSpacing()}")
    
    registration_method = sitk.ImageRegistrationMethod()
    
    # Metric
    registration_method.SetMetricAsMattesMutualInformation(16)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.1)
    
    # Interpolator
    registration_method.SetInterpolator(sitk.sitkLinear)
    
    # Optimizer
    num_iterations = 50
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=num_iterations,
        convergenceMinimumValue=1e-4,
        convergenceWindowSize=5
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()
    
    # Initial transform
    if initial_transform is None:
        initial_transform = sitk.CenteredTransformInitializer(
            fixed_image, moving_image, sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )
    
    registration_method.SetInitialTransform(initial_transform)
    
    # Multi-resolution framework
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[8,4,2])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[4,2,1])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    
    # Add progress callback with tqdm
    progress_callback = RegistrationProgressCallback(num_iterations)
    registration_method.AddCommand(sitk.sitkIterationEvent, progress_callback)
    
    transform = registration_method.Execute(fixed_image, moving_image)
    progress_callback.close()
    
    elapsed_time = time.time() - start_time
    log_progress(f"Rigid registration completed in {elapsed_time:.2f} seconds")
    return transform

def perform_affine_registration(fixed_image, moving_image, initial_transform=None):
    """Perform affine registration with optimized parameters."""
    start_time = time.time()
    
    # Cast images to float32 for registration compatibility
    fixed_image = sitk.Cast(fixed_image, sitk.sitkFloat32)
    moving_image = sitk.Cast(moving_image, sitk.sitkFloat32)
    
    # Log image information
    log_progress(f"Fixed image size: {fixed_image.GetSize()}, spacing: {fixed_image.GetSpacing()}")
    log_progress(f"Moving image size: {moving_image.GetSize()}, spacing: {moving_image.GetSpacing()}")
    
    registration_method = sitk.ImageRegistrationMethod()
    
    # Metric
    registration_method.SetMetricAsMattesMutualInformation(16)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.1)
    
    # Interpolator
    registration_method.SetInterpolator(sitk.sitkLinear)
    
    # Optimizer
    num_iterations = 50
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=num_iterations,
        convergenceMinimumValue=1e-4,
        convergenceWindowSize=5
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()
    
    # Initial transform
    if initial_transform is None:
        initial_transform = sitk.CenteredTransformInitializer(
            fixed_image, moving_image, sitk.AffineTransform(3),
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )
    
    registration_method.SetInitialTransform(initial_transform)
    
    # Multi-resolution framework
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[8,4,2])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[4,2,1])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    
    # Add progress callback with tqdm
    progress_callback = RegistrationProgressCallback(num_iterations)
    registration_method.AddCommand(sitk.sitkIterationEvent, progress_callback)
    
    transform = registration_method.Execute(fixed_image, moving_image)
    progress_callback.close()
    
    elapsed_time = time.time() - start_time
    log_progress(f"Affine registration completed in {elapsed_time:.2f} seconds")
    return transform

def perform_bspline_registration(fixed_image, moving_image, initial_transform=None):
    """Perform BSpline registration for fine alignment with optimized parameters."""
    start_time = time.time()
    
    # Cast images to float32 for registration compatibility
    fixed_image = sitk.Cast(fixed_image, sitk.sitkFloat32)
    moving_image = sitk.Cast(moving_image, sitk.sitkFloat32)
    
    # Log image information
    log_progress(f"Fixed image size: {fixed_image.GetSize()}, spacing: {fixed_image.GetSpacing()}")
    log_progress(f"Moving image size: {moving_image.GetSize()}, spacing: {moving_image.GetSpacing()}")
    
    registration_method = sitk.ImageRegistrationMethod()
    
    # Metric - using fewer bins for faster computation
    registration_method.SetMetricAsMattesMutualInformation(8)  # Reduced from 16
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.05)  # Reduced from 0.1
    
    # Interpolator
    registration_method.SetInterpolator(sitk.sitkLinear)  # Changed from BSpline for speed
    
    # Optimizer - using LBFGSB with more relaxed parameters
    registration_method.SetOptimizerAsLBFGSB(
        gradientConvergenceTolerance=1e-3,  # Relaxed from 1e-4
        numberOfIterations=30,  # Reduced from 50
        maximumNumberOfCorrections=5,  # Added to limit memory usage
        maximumNumberOfFunctionEvaluations=30,  # Added to limit iterations
        costFunctionConvergenceFactor=1e7  # Added to help convergence
    )
    
    # Transform - using coarser mesh for faster computation
    transform_domain_mesh_size = [8, 8, 8]  # Increased from [4,4,4] for faster computation
    bspline_transform = sitk.BSplineTransformInitializer(
        fixed_image, transform_domain_mesh_size
    )
    
    if initial_transform is not None:
        # Combine initial transform with BSpline
        composite_transform = sitk.CompositeTransform(3)
        composite_transform.AddTransform(initial_transform)
        composite_transform.AddTransform(bspline_transform)
        registration_method.SetInitialTransform(composite_transform)
    else:
        registration_method.SetInitialTransform(bspline_transform)
    
    # Multi-resolution framework - more aggressive downsampling
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[8,4,2])  # More aggressive downsampling
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[4,2,1])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    
    # Add progress callback with tqdm
    num_iterations = 30  # Match with optimizer iterations
    progress_callback = RegistrationProgressCallback(num_iterations)
    registration_method.AddCommand(sitk.sitkIterationEvent, progress_callback)
    
    transform = registration_method.Execute(fixed_image, moving_image)
    progress_callback.close()
    
    elapsed_time = time.time() - start_time
    log_progress(f"BSpline registration completed in {elapsed_time:.2f} seconds")
    return transform

def parallel_compute_metric(args):
    """Helper function for parallel metric computation."""
    fixed_path, moving_path, metric_type = args
    try:
        fixed = sitk.ReadImage(str(fixed_path))
        moving = sitk.ReadImage(str(moving_path))
        metric_value = compute_quick_metric(fixed, moving, metric_type)
        return metric_value
    except Exception as e:
        log_progress(f"Error computing metric between {fixed_path} and {moving_path}: {str(e)}", level='error')
        return float('inf')
    finally:
        del fixed, moving

def save_registered_image(image, output_path):
    """Save registered image with progress logging."""
    log_progress(f"Saving registered image to {output_path}...")
    sitk.WriteImage(image, output_path)
    log_progress("✓ Image saved successfully")
