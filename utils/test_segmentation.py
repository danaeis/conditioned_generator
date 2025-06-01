import os
import torch
import numpy as np
import nibabel as nib
import pydicom
from pathlib import Path
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd, ScaleIntensityRanged, EnsureTyped
from monai.networks.nets import DiNTS, TopologyInstance
from monai.inferers import SlidingWindowInferer

# Set memory optimization
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.cuda.empty_cache()

def load_dicom_series(dicom_dir):
    """Load a series of DICOM files and return as a 3D numpy array."""
    dicom_files = sorted(Path(dicom_dir).glob('*.dcm'))
    if not dicom_files:
        raise ValueError(f"No DICOM files found in {dicom_dir}")
    
    # Load first DICOM to get metadata
    first_slice = pydicom.dcmread(str(dicom_files[0]))
    rows = first_slice.Rows
    cols = first_slice.Columns
    num_slices = len(dicom_files)
    
    # Initialize 3D array
    volume = np.zeros((num_slices, rows, cols), dtype=np.float32)
    
    # Load all slices
    for i, dcm_file in enumerate(dicom_files):
        ds = pydicom.dcmread(str(dcm_file))
        volume[i] = ds.pixel_array.astype(np.float32)
    
    # Apply rescale slope and intercept if available
    if hasattr(first_slice, 'RescaleSlope'):
        volume = volume * first_slice.RescaleSlope
    if hasattr(first_slice, 'RescaleIntercept'):
        volume = volume + first_slice.RescaleIntercept
    
    return volume, first_slice

# Load a single test volume from DICOM
test_volume_path = "../ncct_cect/vindr_ds/batches/test_batch/1.2.840.113619.2.278.3.717616.306.1582703645.511/1.2.840.113619.2.278.3.717616.306.1582703645.516.3/"
print(f"Loading test volume from DICOM directory: {test_volume_path}")

# Load and inspect original DICOM data
original_data, first_slice = load_dicom_series(test_volume_path)
print("\nOriginal DICOM image statistics:")
print(f"Shape: {original_data.shape}")
print(f"Value range: [{original_data.min():.3f}, {original_data.max():.3f}]")
print(f"Mean: {original_data.mean():.3f}")
print(f"Std: {original_data.std():.3f}")
print(f"Unique values: {np.unique(original_data)}")

# Create a temporary NIfTI file for MONAI processing
temp_nifti_path = "temp_volume.nii.gz"
temp_img = nib.Nifti1Image(original_data, np.eye(4))
nib.save(temp_img, temp_nifti_path)

# Load and preprocess the volume
preprocessing = Compose([
    LoadImaged(keys="image"),
    EnsureChannelFirstd(keys="image"),
    Orientationd(keys="image", axcodes="RAS"),
    Spacingd(keys="image", pixdim=[1, 1, 1], mode="bilinear"),
    ScaleIntensityRanged(keys="image", a_min=-1024, a_max=3071, b_min=0, b_max=1, clip=True),  # Adjusted for CT HU range
    EnsureTyped(keys="image")
])

# Process the volume
data_dict = {"image": temp_nifti_path}
processed = preprocessing(data_dict)
input_tensor = processed["image"].unsqueeze(0)  # Add batch dimension
print(f"\nPreprocessed input tensor shape: {input_tensor.shape}")
print(f"Preprocessed value range: [{input_tensor.min():.3f}, {input_tensor.max():.3f}]")
print(f"Preprocessed mean: {input_tensor.mean():.3f}")
print(f"Preprocessed std: {input_tensor.std():.3f}")

# Load model architecture and weights
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

bundle_root = "utils/bundles/multi_organ_segmentation"
arch_ckpt_path = os.path.join(bundle_root, "models/search_code_18590.pt")
model_path = os.path.join(bundle_root, "models/model.pt")

print("Loading architecture and model...")
arch_ckpt = torch.load(arch_ckpt_path, map_location=device, weights_only=False)
model_weights = torch.load(model_path, map_location=device)

print("Model weights keys:", list(model_weights.keys())[:5], "...")  # Print first 5 keys

# Create model
dints_space = TopologyInstance(
    channel_mul=1,
    num_blocks=12,
    num_depths=4,
    use_downsample=True,
    arch_code=[arch_ckpt['arch_code_a'], arch_ckpt['arch_code_c']],
    device=device
)

network = DiNTS(
    dints_space=dints_space,
    in_channels=1,
    num_classes=8,
    use_downsample=True,
    node_a=torch.from_numpy(arch_ckpt['node_a'])
).to(device)

# Load weights
network.load_state_dict(model_weights)
network.eval()

# Test a single forward pass first
print("\nTesting single forward pass...")
with torch.no_grad():
    # Take a small patch for testing
    test_patch = input_tensor[:, :, :64, :64, :64].to(device)
    print(f"Test patch shape: {test_patch.shape}")
    print(f"Test patch value range: [{test_patch.min():.3f}, {test_patch.max():.3f}]")
    print(f"Test patch mean: {test_patch.mean():.3f}")
    print(f"Test patch std: {test_patch.std():.3f}")
    
    # Forward pass
    test_output = network(test_patch)
    print(f"Test output shape: {test_output.shape}")
    print(f"Test output value range: [{test_output.min():.3f}, {test_output.max():.3f}]")
    print(f"Test output mean: {test_output.mean():.3f}")
    print(f"Test output std: {test_output.std():.3f}")

# Create inferer with smaller ROI size
inferer = SlidingWindowInferer(
    roi_size=(64, 64, 64),  # Smaller ROI size
    sw_batch_size=1,
    overlap=0.5,
    mode="gaussian",
    padding_mode="constant",
    cval=0.0
)

# Run inference
print("\nRunning full inference...")
with torch.no_grad():
    output = inferer(input_tensor.to(device), network)
    print(f"Raw output shape: {output.shape}")
    print(f"Raw output value range: [{output.min():.3f}, {output.max():.3f}]")
    print(f"Raw output mean: {output.mean():.3f}")
    print(f"Raw output std: {output.std():.3f}")
    
    # Apply softmax
    output_softmax = torch.softmax(output, dim=1)
    print(f"Softmax output value range: [{output_softmax.min():.3f}, {output_softmax.max():.3f}]")
    print(f"Softmax output mean: {output_softmax.mean():.3f}")
    
    # Get class probabilities
    class_probs = output_softmax.mean(dim=(0, 2, 3, 4))
    print(f"Class probabilities: {class_probs.cpu().numpy()}")
    
    pred = torch.argmax(output, dim=1)
    pred_np = pred.cpu().numpy()

print(f"\nFinal prediction shape: {pred_np.shape}")
print(f"Unique labels in prediction: {np.unique(pred_np)}")
print(f"Label distribution: {np.bincount(pred_np.flatten())}")

# Save the prediction
output_dir = "utils/debug/ncct_cect/vindr_ds/test_segmentation"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "test_prediction.nii.gz")

# Save the prediction
pred_img = nib.Nifti1Image(pred_np[0], temp_img.affine, temp_img.header)
nib.save(pred_img, output_path)
print(f"Saved prediction to: {output_path}")

# Clean up temporary file
os.remove(temp_nifti_path) 