# models/autoencoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class PhaseConditionalEncoder(nn.Module):
    def __init__(self, in_channels=1, phase_dim=4):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels + phase_dim, 16, 3, padding=1)
        self.conv2 = nn.Conv3d(16, 32, 3, padding=1)
        self.fc = nn.Linear(32 * 32 * 32 * 32, 256)

    def forward(self, x, phase_vec):
        B, C, D, H, W = x.shape
        phase_map = phase_vec.view(B, -1, 1, 1, 1).expand(-1, -1, D, H, W)
        x = torch.cat([x, phase_map], dim=1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(B, -1)
        return self.fc(x)

class PhaseConditionalDecoder(nn.Module):
    def __init__(self, phase_dim=4):
        super().__init__()
        self.fc = nn.Linear(256 + phase_dim, 32 * 32 * 32 * 32)
        self.deconv1 = nn.ConvTranspose3d(32, 16, 3, padding=1)
        self.deconv2 = nn.ConvTranspose3d(16, 1, 3, padding=1)

    def forward(self, z, phase_vec):
        z = torch.cat([z, phase_vec], dim=1)
        x = F.relu(self.fc(z))
        x = x.view(-1, 32, 32, 32, 32)
        x = F.relu(self.deconv1(x))
        return torch.sigmoid(self.deconv2(x))

class PhaseAutoencoder(nn.Module):
    def __init__(self, in_channels=1, phase_dim=4):
        super().__init__()
        self.encoder = PhaseConditionalEncoder(in_channels, phase_dim)
        self.decoder = PhaseConditionalDecoder(phase_dim)

    def forward(self, x, phase_vec):
        z = self.encoder(x, phase_vec)
        return self.decoder(z, phase_vec)
