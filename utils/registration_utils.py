import SimpleITK as sitk
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import logging
import time
import os

# Set up logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(__file__), 'registration.log'))
    ]
)

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

def register_to_atlas(fixed_image, moving_image, transform_type='multi_step'):
    """
    Enhanced registration with multi-step approach and optimized parameters.
    """
    start_time = time.time()
    
    # Cast images to float32 for registration compatibility
    logging.info("Casting images to float32")
    fixed_image = sitk.Cast(fixed_image, sitk.sitkFloat32)
    moving_image = sitk.Cast(moving_image, sitk.sitkFloat32)
    
    if transform_type == 'multi_step':
        # Step 1: Rigid registration
        logging.info("Starting rigid registration...")
        rigid_transform = perform_rigid_registration(fixed_image, moving_image)
        logging.info("Rigid registration completed")
        
        # Step 2: Affine registration using rigid result as initial transform
        logging.info("Starting affine registration...")
        affine_transform = perform_affine_registration(fixed_image, moving_image, rigid_transform)
        logging.info("Affine registration completed")
        
        # Step 3: BSpline registration (optional, more accurate but slower)
        # logging.info("Starting BSpline registration...")
        # final_transform = perform_bspline_registration(fixed_image, moving_image, affine_transform)
        # logging.info("BSpline registration completed")
        final_transform = affine_transform
        # Resample using final transform
        logging.info("Resampling final image...")
        registered_image = sitk.Resample(moving_image, fixed_image, final_transform,
                                       sitk.sitkBSpline, 0.0, moving_image.GetPixelID())
        logging.info("Resampling completed")
        
        elapsed_time = time.time() - start_time
        logging.info(f"Total registration took {elapsed_time:.2f} seconds")
        return final_transform, registered_image
    else:
        # Fallback to original registration method
        logging.info("Using fallback registration method")
        return perform_affine_registration(fixed_image, moving_image)

def perform_rigid_registration(fixed_image, moving_image, initial_transform=None):
    """Perform rigid registration with optimized parameters."""
    start_time = time.time()
    
    # Cast images to float32 for registration compatibility
    fixed_image = sitk.Cast(fixed_image, sitk.sitkFloat32)
    moving_image = sitk.Cast(moving_image, sitk.sitkFloat32)
    
    registration_method = sitk.ImageRegistrationMethod()
    
    # Metric
    registration_method.SetMetricAsMattesMutualInformation(16)  # Reduced bins
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.1)  # Reduced sampling
    
    # Interpolator
    registration_method.SetInterpolator(sitk.sitkLinear)
    
    # Optimizer
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=50,  # Reduced iterations
        convergenceMinimumValue=1e-4,  # Relaxed convergence
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
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[8,4,2])  # More aggressive downsampling
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[4,2,1])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    
    transform = registration_method.Execute(fixed_image, moving_image)
    elapsed_time = time.time() - start_time
    logging.info(f"Rigid registration took {elapsed_time:.2f} seconds")
    return transform

def perform_affine_registration(fixed_image, moving_image, initial_transform=None):
    """Perform affine registration with optimized parameters."""
    start_time = time.time()
    
    # Cast images to float32 for registration compatibility
    fixed_image = sitk.Cast(fixed_image, sitk.sitkFloat32)
    moving_image = sitk.Cast(moving_image, sitk.sitkFloat32)
    
    registration_method = sitk.ImageRegistrationMethod()
    
    # Metric
    registration_method.SetMetricAsMattesMutualInformation(16)  # Reduced bins
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.1)  # Reduced sampling
    
    # Interpolator
    registration_method.SetInterpolator(sitk.sitkLinear)
    
    # Optimizer
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=50,  # Reduced iterations
        convergenceMinimumValue=1e-4,  # Relaxed convergence
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
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[8,4,2])  # More aggressive downsampling
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[4,2,1])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    
    transform = registration_method.Execute(fixed_image, moving_image)
    elapsed_time = time.time() - start_time
    logging.info(f"Affine registration took {elapsed_time:.2f} seconds")
    return transform

def perform_bspline_registration(fixed_image, moving_image, initial_transform=None):
    """Perform BSpline registration for fine alignment."""
    start_time = time.time()
    
    # Cast images to float32 for registration compatibility
    fixed_image = sitk.Cast(fixed_image, sitk.sitkFloat32)
    moving_image = sitk.Cast(moving_image, sitk.sitkFloat32)
    
    registration_method = sitk.ImageRegistrationMethod()
    
    # Metric
    registration_method.SetMetricAsMattesMutualInformation(16)  # Reduced bins
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.1)  # Reduced sampling
    
    # Interpolator
    registration_method.SetInterpolator(sitk.sitkBSpline)
    
    # Optimizer
    registration_method.SetOptimizerAsLBFGSB(
        gradientConvergenceTolerance=1e-4,  # Relaxed convergence
        numberOfIterations=50  # Reduced iterations
    )
    
    # Transform
    transform_domain_mesh_size = [4, 4, 4]  # Reduced mesh size
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
    
    # Multi-resolution framework
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[8,4,2])  # More aggressive downsampling
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[4,2,1])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    
    transform = registration_method.Execute(fixed_image, moving_image)
    elapsed_time = time.time() - start_time
    logging.info(f"BSpline registration took {elapsed_time:.2f} seconds")
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
        logging.error(f"Error computing metric between {fixed_path} and {moving_path}: {str(e)}")
        return float('inf')
    finally:
        del fixed, moving

def save_registered_image(image, output_path):
    sitk.WriteImage(image, output_path)
