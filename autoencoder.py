import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from modules import Encoder, Decoder, VQEmbedding
from loss import LPIPSWithDiscriminator
from distribution import DiagonalGaussianDistribution

class TAutoencoderKL(pl.LightningModule):
    def __init__(self,
                 latent_dim = None, 
                 *, 
                 image_size=32, 
                 patch_size=4,
                 in_channels=3, 
                 hidden_size=1152, 
                 depth=12, 
                 num_heads=6, 
                 mlp_ratio=4.0, 
                 num_classes=10, 
                 dropout_prob=0.1,
                 z_channels=None,
                 lr=0.0001,
                 disc_start=50001,
                 kl_weight=1.0e-6,
                 num_epochs=8,
                 ):
        super().__init__()
        if latent_dim is None:
            self.latent_dim = (patch_size, patch_size, in_channels)
        else:
            self.latent_dim = latent_dim
        
        if z_channels is None:
            self.z_channels = self.latent_dim[2]
        else:
            self.z_channels = z_channels
        self.image_size = image_size
        self.learning_rate = lr
        assert isinstance(latent_dim, tuple) and len(latent_dim) == 3, 'Latent_dim must be a tuple of length 3 in the form (H, W, C)'
        #self.image_key = image_key
        self.encoder = Encoder(self.latent_dim, image_size=image_size, patch_size=patch_size, in_channels=in_channels, hidden_size=hidden_size, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, num_classes=num_classes, dropout_prob=dropout_prob)
        self.decoder = Decoder(self.latent_dim, image_size=image_size, patch_size=patch_size, in_channels=in_channels, hidden_size=hidden_size, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, num_classes=num_classes, dropout_prob=dropout_prob)
        self.loss = LPIPSWithDiscriminator(disc_start=disc_start, kl_weight=kl_weight, num_epochs=num_epochs)

        self.embed_dim = self.latent_dim[2]
        self.quant_conv = torch.nn.Conv2d(2*self.z_channels, 2*self.embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(self.embed_dim, self.z_channels, 1)
        
        

    def encode(self, x, y):
        h = self.encoder(x, y)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z, y):
        z = self.post_quant_conv(z)
        dec = self.decoder(z, y)
        return dec

    def forward(self, input, label, sample_posterior=True):
        posterior = self.encode(input, label)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z, label)
        return dec, posterior

    def get_input(self, x):
        if len(x.shape) == 3:
            x = x[..., None]
        if x.shape[-1] != self.image_size:
            x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        else:
            x = x.to(memory_format=torch.contiguous_format).float()
        return x


    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return opt_ae, opt_disc

    def get_last_layer(self):
        return self.decoder.final_layer.linear.weight


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
        
        