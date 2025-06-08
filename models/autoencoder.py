# models/autoencoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.skip = nn.Conv3d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x):
        residual = self.skip(x)
        out = F.leaky_relu(self.conv1(x), 0.2)
        out = self.conv2(out)
        out = out + residual
        return F.leaky_relu(out, 0.2)

class PhaseConditionalEncoder(nn.Module):
    def __init__(self, in_channels=1, phase_dim=1):
        super().__init__()
        self.in_channels = in_channels
        self.phase_dim = phase_dim
        
        # Initial convolution
        self.init_conv = nn.Conv3d(in_channels + phase_dim, 16, 3, padding=1)
        
        # Downsampling path with residual blocks
        self.down1 = nn.Sequential(
            nn.Conv3d(16, 32, 4, stride=2, padding=1),
            ResidualBlock(32, 32)
        )
        self.down2 = nn.Sequential(
            nn.Conv3d(32, 64, 4, stride=2, padding=1),
            ResidualBlock(64, 64)
        )
        self.down3 = nn.Sequential(
            nn.Conv3d(64, 128, 4, stride=2, padding=1),
            ResidualBlock(128, 128)
        )
        self.down4 = nn.Sequential(
            nn.Conv3d(128, 256, 4, stride=2, padding=1),
            ResidualBlock(256, 256)
        )
        
        self.fc = None
        self.final_shape = None

    def forward(self, x, phase_condition):
        print("\n=== Encoder Forward Pass ===")
        print(f"Input x shape: {x.shape}, phase_condition shape: {phase_condition.shape}")
        
        # Concatenate input with phase condition
        x = torch.cat([x, phase_condition], dim=1)
        print(f"After concat: {x.shape}")
        
        # Store intermediate features for skip connections
        features = []

        # Initial convolution
        x = F.leaky_relu(self.init_conv(x), 0.2)
        print(f"After init conv: {x.shape}")
        features.append(x)
        
        # Downsampling path
        x = self.down1(x)
        features.append(x)
        print(f"After down1: {x.shape}")
        
        x = self.down2(x)
        features.append(x)
        print(f"After down2: {x.shape}")
        
        x = self.down3(x)
        features.append(x)
        print(f"After down3: {x.shape}")
        
        x = self.down4(x)
        features.append(x)
        print(f"After down4: {x.shape}")
        
        if self.fc is None:
            self.final_shape = x.shape[1:]
            fc_input_size = x.numel() // x.size(0)
            print(f"FC input size: {fc_input_size}")
            self.fc = nn.Linear(fc_input_size, 128).to(x.device)
        
        x = x.view(x.size(0), -1)
        print(f"Flattened shape before FC: {x.shape}")
        x = self.fc(x)
        print(f"Encoder output: {x.shape}")
        print("=== End Encoder ===\n")
        
        return x, features

class PhaseConditionalDecoder(nn.Module):
    def __init__(self, phase_dim=1):
        super().__init__()
        self.phase_dim = phase_dim
        self.deconv_layers = None
        self.fc = None
        self.phase_fc = None
        self.encoder = None
        
        # Upsampling path with residual blocks
        # First upsampling block
        self.up1 = nn.Sequential(
            ResidualBlock(256, 256),
            nn.ConvTranspose3d(256, 128, 4, stride=2, padding=1)  # 17x32x32 -> 35x64x64
        )
        
        # Second upsampling block
        self.up2 = nn.Sequential(
            ResidualBlock(256, 128),  # 256 = 128 + 128 from skip connection
            nn.ConvTranspose3d(128, 64, 4, stride=2, padding=1)  # 35x64x64 -> 70x128x128
        )
        
        # Third upsampling block
        self.up3 = nn.Sequential(
            ResidualBlock(128, 64),  # 128 = 64 + 64 from skip connection
            nn.ConvTranspose3d(64, 32, 4, stride=2, padding=1)  # 70x128x128 -> 140x256x256
        )
        
        # Fourth upsampling block
        self.up4 = nn.Sequential(
            ResidualBlock(64, 32),  # 64 = 32 + 32 from skip connection
            nn.ConvTranspose3d(32, 16, 4, stride=2, padding=1)  # 140x256x256 -> 281x512x512
        )
        
        self.final_conv = nn.Conv3d(16, 1, 3, padding=1)

    def forward(self, z, phase_condition, features):
        print("\n=== Decoder Forward Pass ===")
        print(f"Input z shape: {z.shape}, phase_condition shape: {phase_condition.shape}")
        print(f"Skip connection features shapes: {[f.shape for f in features]}")
        
        B = z.size(0)
        if self.fc is None:
            final_shape = self.encoder.final_shape
            fc_output_size = final_shape[0] * final_shape[1] * final_shape[2] * final_shape[3]
            self.fc = nn.Linear(128, fc_output_size).to(z.device)
            self.phase_fc = nn.Linear(self.phase_dim, final_shape[0]).to(z.device)
        
        # Process latent vector
        x = F.relu(self.fc(z))
        x = x.view(B, *self.encoder.final_shape)
        print(f"After FC and reshape: {x.shape}")
        
        # Process phase condition
        phase_flat = phase_condition[:, :, 0, 0, 0]
        phase_features = self.phase_fc(phase_flat)
        phase_features = phase_features.view(B, -1, 1, 1, 1)
        phase_features = phase_features.expand(B, -1, *x.shape[2:])
        x = x + phase_features
        print(f"After phase features addition: {x.shape}")
        
        # Upsampling path with skip connections
        x = self.up1(x)  # 256 -> 128 channels, 17x32x32 -> 35x64x64
        print(f"After up1: {x.shape}")
        # Skip connection 1 (with features[-2] which is 128x35x64x64)
        skip1 = features[-2]
        if skip1.shape[2:] != x.shape[2:]:
            skip1 = F.interpolate(skip1, size=x.shape[2:], mode='trilinear', align_corners=False)
        x = torch.cat([x, skip1], dim=1)
        print(f"After skip connection 1: {x.shape}")
        
        x = self.up2(x)  # 256 -> 128 channels, 35x64x64 -> 70x128x128
        print(f"After up2: {x.shape}")
        # Skip connection 2 (with features[-3] which is 64x70x128x128)
        skip2 = features[-3]
        if skip2.shape[2:] != x.shape[2:]:
            skip2 = F.interpolate(skip2, size=x.shape[2:], mode='trilinear', align_corners=False)
        x = torch.cat([x, skip2], dim=1)
        print(f"After skip connection 2: {x.shape}")
        
        x = self.up3(x)  # 128 -> 64 channels, 70x128x128 -> 140x256x256
        print(f"After up3: {x.shape}")
        # Skip connection 3 (with features[-4] which is 32x140x256x256)
        skip3 = features[-4]
        if skip3.shape[2:] != x.shape[2:]:
            skip3 = F.interpolate(skip3, size=x.shape[2:], mode='trilinear', align_corners=False)
        x = torch.cat([x, skip3], dim=1)
        print(f"After skip connection 3: {x.shape}")
        
        x = self.up4(x)  # 64 -> 32 channels, 140x256x256 -> 281x512x512
        print(f"After up4: {x.shape}")
        # Skip connection 4 (with features[-5] which is 16x281x512x512)
        skip4 = features[-5]
        if skip4.shape[2:] != x.shape[2:]:
            skip4 = F.interpolate(skip4, size=x.shape[2:], mode='trilinear', align_corners=False)
        x = torch.cat([x, skip4], dim=1)
        print(f"After skip connection 4: {x.shape}")
        
        # Final convolution
        x = self.final_conv(x)
        print(f"After final conv: {x.shape}")
        
        # Ensure output matches input dimensions
        if x.shape[2:] != phase_condition.shape[2:]:
            x = F.interpolate(x, size=phase_condition.shape[2:], mode='trilinear', align_corners=False)
            print(f"After interpolate: {x.shape}")
        
        print("=== End Decoder ===\n")
        return torch.sigmoid(x)

class PhaseAutoencoder(nn.Module):
    def __init__(self, in_channels=1, phase_dim=1):
        super().__init__()
        self.encoder = PhaseConditionalEncoder(in_channels, phase_dim)
        self.decoder = PhaseConditionalDecoder(phase_dim)
        self.decoder.encoder = self.encoder
        self.use_checkpoint = False

    def use_checkpointing(self):
        self.use_checkpoint = True

    def _encoder_forward(self, x, phase_condition):
        print(f"[Autoencoder] Calling encoder_forward")
        return self.encoder(x, phase_condition)

    def _decoder_forward(self, z, phase_condition, features):
        print(f"[Autoencoder] Calling decoder_forward")
        return self.decoder(z, phase_condition, features)

    def forward(self, x, input_phase_condition, output_phase_condition):
        print(f"[Autoencoder] Forward called")
        if not self.use_checkpoint:
            z, features = self.encoder(x, input_phase_condition)
            return self.decoder(z, output_phase_condition, features)

        def custom_encoder_forward(*inputs):
            print(f"[Autoencoder] Checkpointed encoder_forward")
            return self._encoder_forward(*inputs)

        def custom_decoder_forward(*inputs):
            print(f"[Autoencoder] Checkpointed decoder_forward")
            return self._decoder_forward(*inputs)

        z, features = checkpoint(
            custom_encoder_forward,
            x,
            input_phase_condition,
            preserve_rng_state=True,
            use_reentrant=False
        )

        return checkpoint(
            custom_decoder_forward,
            z,
            output_phase_condition,
            features,
            preserve_rng_state=True,
            use_reentrant=False
        )
