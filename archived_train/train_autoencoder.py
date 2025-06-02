import torch
import torch.nn as nn
import torch.optim as optim
import SimpleITK as sitk
import numpy as np
from models.autoencoder import PhaseAutoencoder
from torch.utils.data import Dataset, DataLoader
import os
from pathlib import Path
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm


class PhaseVolumeDataset(Dataset):
    def __init__(self, registered_dir, avg_volumes_dir, labels_path):
        self.registered_dir = Path(registered_dir)
        self.avg_volumes_dir = Path(avg_volumes_dir)
        self.phases = ['non-contrast', 'venous', 'arterial']
        self.phase_to_idx = {phase: idx for idx, phase in enumerate(self.phases)}
        
        # Load phase labels
        self.phase_map = self._load_phase_labels(labels_path)
        
        # Load average volumes for each phase
        self.avg_volumes = {}
        for phase in self.phases:
            vol_path = self.avg_volumes_dir / f"average_{phase}.nii.gz"
            if not vol_path.exists():
                raise FileNotFoundError(f"Average volume file not found: {vol_path}")
            
            try:
                vol = sitk.ReadImage(str(vol_path))
                vol_array = sitk.GetArrayFromImage(vol)
                # Store normalized version
                vol_array = (vol_array - vol_array.min()) / (vol_array.max() - vol_array.min())
                self.avg_volumes[phase] = torch.FloatTensor(vol_array).unsqueeze(0)  # Add channel dimension
                print(f"Successfully loaded {phase} average volume with shape: {vol_array.shape}")
            except Exception as e:
                raise RuntimeError(f"Error loading {phase} average volume from {vol_path}: {str(e)}")
        
        # Get list of registered volumes
        self.volume_paths = []
        for filename in os.listdir(self.registered_dir):
            if not filename.endswith('_registered.nii.gz'):
                continue
                
            # Extract study_id and series_id from filename
            parts = filename.replace('_registered.nii.gz', '').split('_')
            if len(parts) != 2:
                print(f"Skipping {filename} - invalid format")
                continue
                
            study_id, series_id = parts
            phase = self.phase_map.get((study_id, series_id))
            if phase is None:
                print(f"Warning: No phase label found for {filename}")
                continue
                
            self.volume_paths.append((filename, study_id, series_id, phase))
        
        print(f"Found {len(self.volume_paths)} registered volumes")
    
    def _load_phase_labels(self, labels_path):
        """Load phase labels from CSV file."""
        df = pd.read_csv(labels_path)
        phase_map = {}
        for _, row in df.iterrows():
            study_id = row['StudyInstanceUID']
            series_id = row['SeriesInstanceUID']
            phase = row['Label'].lower()
            phase_map[(study_id, series_id)] = phase
        return phase_map
    
    def __len__(self):
        return len(self.volume_paths)
    
    def __getitem__(self, idx):
        filename, study_id, series_id, input_phase = self.volume_paths[idx]
        
        # Load and preprocess input volume
        vol_path = self.registered_dir / filename
        vol = sitk.ReadImage(str(vol_path))
        vol_array = sitk.GetArrayFromImage(vol)
        
        # Normalize to [0, 1]
        vol_array = (vol_array - vol_array.min()) / (vol_array.max() - vol_array.min())
        input_volume = torch.FloatTensor(vol_array).unsqueeze(0)  # Add channel dimension
        
        # Get average volume for input phase and ensure it matches input dimensions
        input_avg_volume = self.avg_volumes[input_phase]
        if input_avg_volume.shape != input_volume.shape:
            # Resample average volume to match input dimensions
            avg_vol = sitk.GetImageFromArray(input_avg_volume.squeeze(0).numpy())
            resampler = sitk.ResampleImageFilter()
            resampler.SetSize(vol.GetSize())
            resampler.SetOutputSpacing(vol.GetSpacing())
            resampler.SetOutputOrigin(vol.GetOrigin())
            resampler.SetOutputDirection(vol.GetDirection())
            resampler.SetDefaultPixelValue(vol.GetPixelIDValue())
            avg_vol = resampler.Execute(avg_vol)
            input_avg_volume = torch.FloatTensor(sitk.GetArrayFromImage(avg_vol)).unsqueeze(0)
        
        # Calculate difference between input volume and its phase average
        diff = input_volume - input_avg_volume
        diff = (diff - diff.min()) / (diff.max() - diff.min())
        
        # Create dictionary with all phase conditions
        phase_conditions = {}
        for phase in self.phases:
            phase_avg = self.avg_volumes[phase]
            if phase_avg.shape != input_volume.shape:
                # Resample phase average to match input dimensions
                avg_vol = sitk.GetImageFromArray(phase_avg.squeeze(0).numpy())
                resampler = sitk.ResampleImageFilter()
                resampler.SetSize(vol.GetSize())
                resampler.SetOutputSpacing(vol.GetSpacing())
                resampler.SetOutputOrigin(vol.GetOrigin())
                resampler.SetOutputDirection(vol.GetDirection())
                resampler.SetDefaultPixelValue(vol.GetPixelIDValue())
                avg_vol = resampler.Execute(avg_vol)
                phase_conditions[phase] = torch.FloatTensor(sitk.GetArrayFromImage(avg_vol)).unsqueeze(0)
            else:
                phase_conditions[phase] = phase_avg
        
        return {
            'input_volume': diff,  # Difference between input volume and its phase average
            'input_phase': input_phase,
            'raw_volume': input_volume,  # Original normalized input volume
            'input_avg_volume': input_avg_volume,  # Average volume of input phase
            'phase_conditions': phase_conditions,  # All phase average volumes for conditioning
            'study_id': study_id,
            'series_id': series_id,
            'original_size': vol.GetSize(),  # Store original size for reference
            'original_spacing': vol.GetSpacing()  # Store original spacing for reference
        }

class Discriminator(nn.Module):
    def __init__(self, in_channels=1, phase_dim=1):
        super().__init__()
        # Initial size: (1, 128, 128, 128)
        self.conv1 = nn.Conv3d(in_channels + phase_dim, 32, 4, stride=2, padding=1)  # -> (32, 64, 64, 64)
        self.conv2 = nn.Conv3d(32, 64, 4, stride=2, padding=1)  # -> (64, 32, 32, 32)
        self.conv3 = nn.Conv3d(64, 128, 4, stride=2, padding=1)  # -> (128, 16, 16, 16)
        self.conv4 = nn.Conv3d(128, 256, 4, stride=2, padding=1)  # -> (256, 8, 8, 8)
        self.fc = nn.Linear(256 * 8 * 8 * 8, 1)
        
    def forward(self, x, phase_condition):
        B, C, D, H, W = x.shape
        phase_map = phase_condition.view(B, -1, 1, 1, 1).expand(-1, -1, D, H, W)
        x = torch.cat([x, phase_map], dim=1)
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = F.leaky_relu(self.conv4(x), 0.2)
        x = x.view(B, -1)
        return torch.sigmoid(self.fc(x))

def visualize_batch(batch, output, epoch, batch_idx, save_dir='visualizations'):
    """Visualize a batch of data and model outputs."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Get data from batch
    input_volume = batch['input_volume'].cpu().numpy()
    input_phase = batch['input_phase'][0]  # Get the first (and only) item since batch_size=1
    phase_condition = batch['phase_conditions'][input_phase].cpu().numpy()
    output_volume = output.detach().cpu().numpy()
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    fig.suptitle(f'Epoch {epoch}, Batch {batch_idx}\nPhase: {input_phase}')
    
    # Function to plot middle slice of a volume
    def plot_slice(ax, volume, title, cmap='gray'):
        middle_slice = volume[0, 0, volume.shape[2]//2, :, :]
        im = ax.imshow(middle_slice, cmap=cmap)
        ax.set_title(title)
        plt.colorbar(im, ax=ax)
    
    # Plot input volume
    plot_slice(axes[0, 0], input_volume, 'Input Volume')
    
    # Plot phase condition
    plot_slice(axes[0, 1], phase_condition, 'Phase Condition (Average Volume)')
    
    # Plot generated output
    plot_slice(axes[1, 0], output_volume, 'Generated Output')
    
    # Plot difference
    diff = np.abs(input_volume - output_volume)
    plot_slice(axes[1, 1], diff, 'Absolute Difference', cmap='hot')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'epoch_{epoch}_batch_{batch_idx}.png'))
    plt.close()

def plot_training_curves(epoch_d_losses, epoch_ae_losses, epoch, save_dir='visualizations'):
    """Plot training curves for discriminator and autoencoder losses."""
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 5))
    plt.plot(epoch_d_losses, label='Discriminator Loss')
    plt.plot(epoch_ae_losses, label='Autoencoder Loss')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title(f'Training Curves - Epoch {epoch}')
    plt.legend()
    plt.savefig(os.path.join(save_dir, f'losses_epoch_{epoch}.png'))
    plt.close()

def train_autoencoder(
    data_dir,
    output_dir,
    avg_volumes_dir,
    labels_path,
    batch_size=1,
    num_epochs=100,
    learning_rate=1e-4,
    device='cuda',
    use_checkpointing=True
):
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize model
    model = PhaseAutoencoder(in_channels=1, phase_dim=1).to(device)
    if use_checkpointing:
        model.use_checkpointing()
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Initialize loss function
    criterion = nn.MSELoss()
    
    # Initialize tensorboard
    writer = SummaryWriter(os.path.join(output_dir, 'logs'))
    
    # Initialize dataset
    dataset = PhaseVolumeDataset(data_dir, avg_volumes_dir, labels_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')):
            # Get input volume and its phase
            input_volume = batch['input_volume'].to(device)  # Difference between input and its phase average
            input_phase = batch['input_phase'][0]
            raw_volume = batch['raw_volume'].to(device)  # Original normalized input volume
            phase_conditions = {phase: cond.to(device) for phase, cond in batch['phase_conditions'].items()}
            
            # For each target phase (excluding input phase)
            for target_phase in dataset.phases:
                if target_phase == input_phase:
                    continue
                
                # Forward pass
                optimizer.zero_grad()
                reconstructed = model(input_volume, phase_conditions[target_phase])
                
                # Calculate loss between reconstructed volume and target phase average
                loss = criterion(reconstructed, phase_conditions[target_phase])
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                # Log to tensorboard
                writer.add_scalar(f'Loss/{input_phase}_to_{target_phase}', loss.item(), 
                                epoch * len(dataloader) + batch_idx)
        
        # Calculate average loss for epoch
        avg_loss = total_loss / (len(dataloader) * (len(dataset.phases) - 1))  # -1 because we skip input phase
        print(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.6f}')
        
        # Log average loss to tensorboard
        writer.add_scalar('Loss/train_avg', avg_loss, epoch)
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(output_dir, 'final_model.pt'))
    writer.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing registered volume files')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save model checkpoints')
    parser.add_argument('--avg_volumes_dir', type=str, required=True, help='Directory containing phase average volumes')
    parser.add_argument('--labels_path', type=str, required=True, help='Path to CSV file containing phase labels')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--use_checkpointing', action='store_true', help='Use gradient checkpointing')
    args = parser.parse_args()
    
    train_autoencoder(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        avg_volumes_dir=args.avg_volumes_dir,
        labels_path=args.labels_path,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        device=args.device,
        use_checkpointing=args.use_checkpointing
    )