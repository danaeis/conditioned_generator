import SimpleITK as sitk

def register_to_atlas(fixed_image, moving_image, transform_type='rigid'):
    """
    Registers moving_image to fixed_image using specified transform.
    Supported types: 'rigid', 'affine'.
    Returns the final transform and the registered image.
    """
    if transform_type not in ['rigid', 'affine']:
        raise ValueError("transform_type must be 'rigid' or 'affine'")

    registration_method = sitk.ImageRegistrationMethod()

    # Metric
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.2)

    # Interpolator
    registration_method.SetInterpolator(sitk.sitkLinear)

    # Optimizer
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100,
                                                       convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Initial Transform
    if transform_type == 'rigid':
        initial_transform = sitk.CenteredTransformInitializer(fixed_image,
                                                              moving_image,
                                                              sitk.Euler3DTransform(),
                                                              sitk.CenteredTransformInitializerFilter.GEOMETRY)
    elif transform_type == 'affine':
        initial_transform = sitk.CenteredTransformInitializer(fixed_image,
                                                              moving_image,
                                                              sitk.AffineTransform(3),
                                                              sitk.CenteredTransformInitializerFilter.GEOMETRY)

    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    # Multi-resolution framework
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Cast both images to Float32 for registration compatibility
    fixed_image = sitk.Cast(fixed_image, sitk.sitkFloat32)
    moving_image = sitk.Cast(moving_image, sitk.sitkFloat32)

    # Run registration
    final_transform = registration_method.Execute(fixed_image, moving_image)

    # Resample moving image to fixed image space
    registered_image = sitk.Resample(moving_image, fixed_image, final_transform,
                                     sitk.sitkLinear, 0.0, moving_image.GetPixelID())

    return final_transform, registered_image

def save_registered_image(image, output_path):
    sitk.WriteImage(image, output_path)
