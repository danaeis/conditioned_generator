# models/autoencoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class PhaseConditionalEncoder(nn.Module):
    def __init__(self, in_channels=1, phase_dim=1):
        super().__init__()
        # Initial size: (1, 128, 128, 128)
        self.conv1 = nn.Conv3d(in_channels + phase_dim, 32, 4, stride=2, padding=1)  # -> (32, 64, 64, 64)
        self.conv2 = nn.Conv3d(32, 64, 4, stride=2, padding=1)  # -> (64, 32, 32, 32)
        self.conv3 = nn.Conv3d(64, 128, 4, stride=2, padding=1)  # -> (128, 16, 16, 16)
        self.conv4 = nn.Conv3d(128, 256, 4, stride=2, padding=1)  # -> (256, 8, 8, 8)
        self.fc = nn.Linear(256 * 8 * 8 * 8, 256)

    def forward(self, x, phase_condition):
        B, C, D, H, W = x.shape
        # Concatenate input with phase condition volume
        x = torch.cat([x, phase_condition], dim=1)
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = F.leaky_relu(self.conv4(x), 0.2)
        x = x.view(B, -1)
        return self.fc(x)

class PhaseConditionalDecoder(nn.Module):
    def __init__(self, phase_dim=1):
        super().__init__()
        self.fc = nn.Linear(256 + phase_dim * 8 * 8 * 8, 256 * 8 * 8 * 8)
        # Initial size: (256, 8, 8, 8)
        self.deconv1 = nn.ConvTranspose3d(256, 128, 4, stride=2, padding=1)  # -> (128, 16, 16, 16)
        self.deconv2 = nn.ConvTranspose3d(128, 64, 4, stride=2, padding=1)  # -> (64, 32, 32, 32)
        self.deconv3 = nn.ConvTranspose3d(64, 32, 4, stride=2, padding=1)  # -> (32, 64, 64, 64)
        self.deconv4 = nn.ConvTranspose3d(32, 1, 4, stride=2, padding=1)  # -> (1, 128, 128, 128)

    def forward(self, z, phase_condition):
        # Reshape phase condition to match latent space
        B = z.size(0)
        phase_flat = phase_condition.view(B, -1)
        z = torch.cat([z, phase_flat], dim=1)
        x = F.relu(self.fc(z))
        x = x.view(-1, 256, 8, 8, 8)
        x = F.leaky_relu(self.deconv1(x), 0.2)
        x = F.leaky_relu(self.deconv2(x), 0.2)
        x = F.leaky_relu(self.deconv3(x), 0.2)
        return torch.sigmoid(self.deconv4(x))

class PhaseAutoencoder(nn.Module):
    def __init__(self, in_channels=1, phase_dim=1):
        super().__init__()
        self.encoder = PhaseConditionalEncoder(in_channels, phase_dim)
        self.decoder = PhaseConditionalDecoder(phase_dim)

    def forward(self, x, phase_condition):
        z = self.encoder(x, phase_condition)
        return self.decoder(z, phase_condition)
