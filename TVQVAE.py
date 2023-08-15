import torch
import numpy as np
from torch import nn as nn
from modules import Encoder, Decoder, VQEmbedding

class TVQVAE(nn.Module):
    def __init__(self, latent_dim = None, *, image_size=32, patch_size=4,in_channels=1, hidden_size=1152, depth=12, num_heads=6, mlp_ratio=4.0, num_classes=10, dropout_prob=0.1):
        super().__init__()
        if latent_dim is None:
            self.latent_dim = (patch_size, patch_size, in_channels)
        else:
            self.latent_dim = latent_dim
        assert isinstance(latent_dim, tuple) and len(latent_dim) == 3, 'Latent_dim must be a tuple of length 3 in the form (H, W, C)'
        D = latent_dim[2]
        self.codebook = VQEmbedding(512, D)
        self.encoder = Encoder(self.latent_dim, image_size=image_size, patch_size=patch_size, in_channels=in_channels, hidden_size=hidden_size, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, num_classes=num_classes, dropout_prob=dropout_prob)
        self.decoder = Decoder(self.latent_dim, image_size=image_size, patch_size=patch_size, in_channels=in_channels, hidden_size=hidden_size, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, num_classes=num_classes, dropout_prob=dropout_prob)
    
    def encode(self, x, y):
        z_e_x = self.encoder(x, y)
        latents = self.codebook(z_e_x)
        return latents
    
    def decode(self, latents, y):
        z_q_x = self.codebook.embedding(latents).permute(0, 3, 1, 2)  # (B, D, H, W)
        x_tilde = self.decoder(z_q_x, y)
        return x_tilde
        
    def forward(self, img, y):
        z_e_x = self.encoder(img, y)
        z_q_x_st, z_q_x = self.codebook.straight_through(z_e_x)
        x_tilde = self.decoder(z_q_x_st, y)
        return x_tilde, z_e_x, z_q_x
        
        