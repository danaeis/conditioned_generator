import torch

model_path = "utils/bundles/multi_organ_segmentation/models/model.pt"

ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
print(f"Type of loaded checkpoint: {type(ckpt)}")
if isinstance(ckpt, dict):
    print(f"Keys in checkpoint: {list(ckpt.keys())}")
else:
    print("Checkpoint is not a dict. Printing object:")
    print(ckpt) 