import os
import json
import torch
import numpy as np
import nibabel as nib
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd, ScaleIntensityRanged, EnsureTyped
from monai.networks.nets import DiNTS, TopologyInstance

# Load the dataset config
bundle_root = "utils/bundles/multi_organ_segmentation"
dataset_dir = "utils/debug/ncct_cect/vindr_ds/original_volumes/"
data_list_file_path = os.path.join(bundle_root, "configs/dataset_0.json")

with open(data_list_file_path, 'r') as f:
    dataset_config = json.load(f)

# Get the first image path
image_path = os.path.join(dataset_dir, dataset_config['testing'][0]['image'])
print(f"Loading image from: {image_path}")

# Load the original image
original_img = nib.load(image_path)
original_data = original_img.get_fdata()
print(f"Original image shape: {original_data.shape}")
print(f"Original image value range: [{original_data.min()}, {original_data.max()}]")

# Create preprocessing pipeline
preprocessing = Compose([
    LoadImaged(keys="image"),
    EnsureChannelFirstd(keys="image"),
    Orientationd(keys="image", axcodes="RAS"),
    Spacingd(keys="image", pixdim=[1, 1, 1], mode="bilinear"),
    ScaleIntensityRanged(keys="image", a_min=-500, a_max=500, b_min=0, b_max=1, clip=True),
    EnsureTyped(keys="image")
])

# Apply preprocessing
data_dict = {"image": image_path}
processed = preprocessing(data_dict)
processed_data = processed["image"].numpy()
print(f"Processed image shape: {processed_data.shape}")
print(f"Processed image value range: [{processed_data.min()}, {processed_data.max()}]")

# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

arch_ckpt_path = os.path.join(bundle_root, "models/search_code_18590.pt")
model_path = os.path.join(bundle_root, "models/model.pt")

print(f"Loading architecture from: {arch_ckpt_path}")
print(f"Loading model weights from: {model_path}")

# Add numpy to safe globals
torch.serialization.add_safe_globals(['numpy.core.multiarray._reconstruct'])

# Load architecture and model with weights_only=False
arch_ckpt = torch.load(arch_ckpt_path, map_location=device, weights_only=False)
model_weights = torch.load(model_path, map_location=device, weights_only=False)

# Create the model architecture
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

# Load the weights
network.load_state_dict(model_weights['model'])

print("Model loaded successfully")

# Test inference
network.eval()
with torch.no_grad():
    input_tensor = torch.from_numpy(processed_data).unsqueeze(0).to(device)
    output = network(input_tensor)
    pred = torch.argmax(output, dim=1)
    pred_np = pred.cpu().numpy()
    print(f"Prediction shape: {pred_np.shape}")
    print(f"Unique labels in prediction: {np.unique(pred_np)}") 