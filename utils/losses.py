# # utils/losses.py
# import torch.nn as nn

# bce_loss = nn.BCELoss()

# def reconstruction_loss(pred, target):
#     return nn.MSELoss()(pred, target)

# def discriminator_loss(real_pred, fake_pred):
#     real_loss = bce_loss(real_pred, torch.ones_like(real_pred))
#     fake_loss = bce_loss(fake_pred, torch.zeros_like(fake_pred))
#     return (real_loss + fake_loss) / 2

# def generator_loss(fake_pred):
#     return bce_loss(fake_pred, torch.ones_like(fake_pred))
