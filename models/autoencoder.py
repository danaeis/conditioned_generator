# models/autoencoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

class PhaseConditionalEncoder(nn.Module):
    def __init__(self, in_channels=1, phase_dim=1):
        super().__init__()
        self.in_channels = in_channels
        self.phase_dim = phase_dim
        
        # Modified conv layers to maintain better spatial information
        self.conv_layers = nn.ModuleList([
            # Input: (B, 2, D, H, W) -> (B, 16, D/2, H/2, W/2)
            nn.Conv3d(in_channels + phase_dim, 16, 4, stride=2, padding=1),
            # Input: (B, 16, D/2, H/2, W/2) -> (B, 32, D/4, H/4, W/4)
            nn.Conv3d(16, 32, 4, stride=2, padding=1),
            # Input: (B, 32, D/4, H/4, W/4) -> (B, 64, D/8, H/8, W/8)
            nn.Conv3d(32, 64, 4, stride=2, padding=1),
            # Input: (B, 64, D/8, H/8, W/8) -> (B, 128, D/16, H/16, W/16)
            nn.Conv3d(64, 128, 4, stride=2, padding=1)
        ])
        
        self.fc = None
        self.final_shape = None

    def _calculate_fc_input_size(self, x):
        for conv in self.conv_layers:
            x = F.leaky_relu(conv(x), 0.2)
        self.final_shape = x.shape[1:]  # Store shape (C, D/32, H/32, W/32)
        return x.numel() // x.size(0)

    def forward(self, x, phase_condition):
        B, C, D, H, W = x.shape
        
        if self.fc is None:
            dummy_input = torch.zeros(1, self.in_channels + self.phase_dim, D, H, W, device=x.device)
            fc_input_size = self._calculate_fc_input_size(dummy_input)
            # Reduced latent space dimension
            self.fc = nn.Linear(fc_input_size, 128).to(x.device)
        
        x = torch.cat([x, phase_condition], dim=1)
        
        for conv in self.conv_layers:
            x = F.leaky_relu(conv(x), 0.2)
        
        x = x.view(B, -1)
        return self.fc(x)

class PhaseConditionalDecoder(nn.Module):
    def __init__(self, phase_dim=1):
        super().__init__()
        self.phase_dim = phase_dim
        self.deconv_layers = None
        self.fc = None
        self.input_shape = None

    def _create_deconv_layers(self, final_shape):
        # Reduced number of channels
        channels = [128, 64, 32, 16, 1]  # Channel progression
        layers = []
        
        for i in range(len(channels)-1):
            # All layers: 2x upsampling with output padding to handle odd dimensions
            layers.append(nn.ConvTranspose3d(
                channels[i], 
                channels[i+1], 
                4, 
                stride=2, 
                padding=1,
                output_padding=1
            ))
        
        return nn.ModuleList(layers)

    def forward(self, z, phase_condition):
        B = z.size(0)
        
        if self.deconv_layers is None:
            final_shape = self.encoder.final_shape
            self.deconv_layers = self._create_deconv_layers(final_shape).to(z.device)
            
            fc_output_size = 128 * final_shape[0] * final_shape[1] * final_shape[2]
            self.fc = nn.Linear(128, fc_output_size).to(z.device)
            self.phase_fc = nn.Linear(self.phase_dim, 128).to(z.device)
        
        # Extract phase value from the 5D tensor (B, 1, D, H, W) -> (B, 1)
        phase_flat = phase_condition[:, :, 0, 0, 0]
        phase_features = self.phase_fc(phase_flat)
        
        x = F.relu(self.fc(z))
        x = x.view(-1, 128, *self.encoder.final_shape[1:])
        
        # Expand phase features to match the current spatial dimensions of x
        phase_features = phase_features.view(B, 128, 1, 1, 1)
        phase_features = phase_features.expand_as(x)
        x = x + phase_features
        
        for i, deconv in enumerate(self.deconv_layers[:-1]):
            x = F.leaky_relu(deconv(x), 0.2)
        
        # Final layer with sigmoid activation
        x = self.deconv_layers[-1](x)
        
        # Ensure output matches input dimensions
        if x.shape[2:] != phase_condition.shape[2:]:
            x = F.interpolate(x, size=phase_condition.shape[2:], mode='trilinear', align_corners=False)
        
        return torch.sigmoid(x)

class PhaseAutoencoder(nn.Module):
    def __init__(self, in_channels=1, phase_dim=1):
        super().__init__()
        self.encoder = PhaseConditionalEncoder(in_channels, phase_dim)
        self.decoder = PhaseConditionalDecoder(phase_dim)
        self.decoder.encoder = self.encoder  # Share encoder reference for shape information
        self.use_checkpoint = False

    def use_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency."""
        self.use_checkpoint = True

    def forward(self, x, input_phase_condition, output_phase_condition):
        """
        Args:
            x: Input volume (B, 1, D, H, W)
            input_phase_condition: Phase condition for input (B, 1, D, H, W)
            output_phase_condition: Phase condition for output (B, 1, D, H, W)
        """
        # Ensure inputs require gradients
        if not x.requires_grad:
            x.requires_grad_(True)
        if not input_phase_condition.requires_grad:
            input_phase_condition.requires_grad_(True)
        if not output_phase_condition.requires_grad:
            output_phase_condition.requires_grad_(True)

        if self.use_checkpoint:
            # Use gradient checkpointing for encoder with explicit use_reentrant=False
            z = checkpoint(
                self.encoder, 
                x, 
                input_phase_condition,
                use_reentrant=False
            )
            # Use gradient checkpointing for decoder with explicit use_reentrant=False
            return checkpoint(
                self.decoder, 
                z, 
                output_phase_condition,
                use_reentrant=False
            )
        else:
            z = self.encoder(x, input_phase_condition)
            return self.decoder(z, output_phase_condition)
