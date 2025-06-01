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

class PhaseVolumeDataset(Dataset):
    def __init__(self, avg_volumes_dir, target_size=(128, 128, 128), use_difference=True):
        self.avg_volumes_dir = Path(avg_volumes_dir)
        self.phases = ['non-contrast', 'venous']
        self.phase_to_idx = {phase: idx for idx, phase in enumerate(self.phases)}
        self.target_size = target_size
        self.use_difference = use_difference
        
        # Load average volumes
        self.volumes = {}
        self.avg_volumes = {}
        for phase in self.phases:
            vol_path = self.avg_volumes_dir / f"average_{phase}.nii.gz"
            if not vol_path.exists():
                raise FileNotFoundError(f"Average volume file not found: {vol_path}")
            
            try:
                vol = sitk.ReadImage(str(vol_path))
                # Resize to target size
                resampler = sitk.ResampleImageFilter()
                resampler.SetSize(self.target_size)
                resampler.SetOutputSpacing([1, 1, 1])
                resampler.SetOutputOrigin([0, 0, 0])
                resampler.SetOutputDirection(vol.GetDirection())
                resampler.SetDefaultPixelValue(vol.GetPixelIDValue())
                vol = resampler.Execute(vol)
                
                vol_array = sitk.GetArrayFromImage(vol)
                # Store both raw and normalized versions
                self.avg_volumes[phase] = torch.FloatTensor(vol_array).unsqueeze(0)  # Add channel dimension
                
                # Normalize to [0, 1]
                vol_array = (vol_array - vol_array.min()) / (vol_array.max() - vol_array.min())
                self.volumes[phase] = torch.FloatTensor(vol_array).unsqueeze(0)  # Add channel dimension
                print(f"Successfully loaded {phase} volume with shape: {vol_array.shape}")
                print(f"Value range: [{vol_array.min():.3f}, {vol_array.max():.3f}]")
            except Exception as e:
                raise RuntimeError(f"Error loading {phase} volume from {vol_path}: {str(e)}")
    
    def __len__(self):
        return len(self.phases)
    
    def __getitem__(self, idx):
        input_phase = self.phases[idx]
        input_volume = self.volumes[input_phase]
        avg_volume = self.avg_volumes[input_phase]
        
        if self.use_difference:
            # Normalize the difference to [0, 1] range
            diff = input_volume - avg_volume
            diff = (diff - diff.min()) / (diff.max() - diff.min())
            return {
                'input_volume': diff,
                'phase_condition': avg_volume,  # Use average volume as condition
                'input_phase': input_phase,
                'raw_volume': input_volume,
                'avg_volume': avg_volume
            }
        else:
            return {
                'input_volume': input_volume,
                'phase_condition': avg_volume,  # Use average volume as condition
                'input_phase': input_phase,
                'raw_volume': input_volume,
                'avg_volume': avg_volume
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
    phase_condition = batch['phase_condition'].cpu().numpy()
    output_volume = output.detach().cpu().numpy()
    input_phase = batch['input_phase']
    
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

def train_autoencoder():
    # Initialize models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    autoencoder = PhaseAutoencoder(in_channels=1, phase_dim=1).to(device)
    discriminator = Discriminator(in_channels=1, phase_dim=1).to(device)
    
    # Initialize optimizers
    ae_optimizer = optim.Adam(autoencoder.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    # Loss functions
    reconstruction_loss = nn.MSELoss()
    adversarial_loss = nn.BCELoss()
    
    # Dataset and dataloader
    avg_volumes_dir = "/media/disk1/saeedeh_danaei/conditioned_generator/utils/debug/ncct_cect/vindr_ds/average_volumes"
    print(f"Loading average volumes from: {avg_volumes_dir}")
    
    try:
        # Create two datasets: one for raw volumes and one for differences
        raw_dataset = PhaseVolumeDataset(avg_volumes_dir, use_difference=False)
        diff_dataset = PhaseVolumeDataset(avg_volumes_dir, use_difference=True)
        
        print("\nRaw Volume Statistics:")
        for phase in raw_dataset.phases:
            vol = raw_dataset.volumes[phase]
            print(f"{phase}: min={vol.min():.3f}, max={vol.max():.3f}, mean={vol.mean():.3f}")
        
        print("\nDifference Volume Statistics:")
        for phase in diff_dataset.phases:
            vol = diff_dataset.volumes[phase]
            print(f"{phase}: min={vol.min():.3f}, max={vol.max():.3f}, mean={vol.mean():.3f}")
        
        # Choose which dataset to use
        dataset = diff_dataset  # or raw_dataset
        print(f"\nUsing {'difference' if dataset.use_difference else 'raw'} volumes for training")
        
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return
    
    num_epochs = 100
    for epoch in range(num_epochs):
        epoch_d_losses = []
        epoch_ae_losses = []
        
        for batch_idx, batch in enumerate(dataloader):
            try:
                input_volume = batch['input_volume'].to(device)
                phase_condition = batch['phase_condition'].to(device)
                input_phase = batch['input_phase']
                
                # Train discriminator
                d_optimizer.zero_grad()
                
                # Real samples
                real_labels = torch.ones(input_volume.size(0), 1).to(device)
                d_real_output = discriminator(input_volume, phase_condition)
                d_real_loss = adversarial_loss(d_real_output, real_labels)
                
                # Generated samples
                fake_volume = autoencoder(input_volume, phase_condition)
                fake_labels = torch.zeros(input_volume.size(0), 1).to(device)
                d_fake_output = discriminator(fake_volume.detach(), phase_condition)
                d_fake_loss = adversarial_loss(d_fake_output, fake_labels)
                
                d_loss = d_real_loss + d_fake_loss
                d_loss.backward()
                d_optimizer.step()
                
                # Train autoencoder
                ae_optimizer.zero_grad()
                
                # Reconstruction loss
                recon_loss = reconstruction_loss(fake_volume, input_volume)
                
                # Adversarial loss for generator
                g_output = discriminator(fake_volume, phase_condition)
                g_loss = adversarial_loss(g_output, real_labels)
                
                # Total loss
                ae_loss = recon_loss + 0.1 * g_loss
                ae_loss.backward()
                ae_optimizer.step()
                
                epoch_d_losses.append(d_loss.item())
                epoch_ae_losses.append(ae_loss.item())
                
                # Visualize every 10 batches
                if batch_idx % 10 == 0:
                    print(f"Epoch [{epoch}/{num_epochs}] Batch [{batch_idx}] "
                          f"D_loss: {d_loss.item():.4f} AE_loss: {ae_loss.item():.4f}")
                    visualize_batch(batch, fake_volume, epoch, batch_idx)
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {str(e)}")
                continue
        
        # Plot training curves at the end of each epoch
        plot_training_curves(epoch_d_losses, epoch_ae_losses, epoch)
        
        # Print epoch statistics
        avg_d_loss = sum(epoch_d_losses) / len(epoch_d_losses)
        avg_ae_loss = sum(epoch_ae_losses) / len(epoch_ae_losses)
        print(f"Epoch [{epoch}/{num_epochs}] Average D_loss: {avg_d_loss:.4f} Average AE_loss: {avg_ae_loss:.4f}")
        
        # Save model checkpoints periodically
        if epoch % 10 == 0:
            try:
                checkpoint_path = f'checkpoints/autoencoder_epoch_{epoch}.pt'
                torch.save({
                    'epoch': epoch,
                    'autoencoder_state_dict': autoencoder.state_dict(),
                    'discriminator_state_dict': discriminator.state_dict(),
                    'ae_optimizer_state_dict': ae_optimizer.state_dict(),
                    'd_optimizer_state_dict': d_optimizer.state_dict(),
                }, checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")
            except Exception as e:
                print(f"Error saving checkpoint: {str(e)}")

if __name__ == "__main__":
    try:
        os.makedirs('checkpoints', exist_ok=True)
        os.makedirs('visualizations', exist_ok=True)
        train_autoencoder()
    except Exception as e:
        print(f"Training failed: {str(e)}") 