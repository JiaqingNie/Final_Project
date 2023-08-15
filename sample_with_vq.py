# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import os
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
import torch.distributed as dist
from diffusion import create_diffusion
import torch.multiprocessing as mp
#from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models
import argparse
from time import time
from TVQVAE import TVQVAE

timestamp = int(time())
 
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12357'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def sample(rank, world_size):
    model = "DiT-B/2"
    vae = 'mse'
    num_classes = 3
    cfg_scale = 4.0
    num_sampling_steps = 1000
    seed = torch.randint(0, 2 ** 32, (1,)).item()
    ckpt = '/project/DiT/results/001-DiT-B-2/checkpoints/0562500.pt'
    # Setup PyTorch:
    setup(rank, world_size)
    torch.manual_seed(seed)
    torch.set_grad_enabled(False)
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)

    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    
    # Load model:
    latent_size = 16
    model = DiT_models[model](
        input_size=latent_size,
        num_classes=num_classes
    ).to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    
    ckpt_path = ckpt 
    print(f"Loading diffusion model from: {ckpt_path}...")
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    print("Loaded diffusion model.")
    model.eval()  # important!
    diffusion = create_diffusion(str(num_sampling_steps))
    vae_path = "/project/TVQVAE/results/1691516614/TVQVAE.pt"
    print("Loading VAE model...")
    vae = torch.load(vae_path)
    vae = vae.to(device)
    print("Loaded VAE model.")

    # Labels to condition the model with (feel free to change):
    class_labels = []
    for i in range(num_classes):
        for _ in range(10):
            class_labels.append(i)

    # Create sampling noise:
    n = len(class_labels)
    z = torch.randn(n, 3, latent_size, latent_size, device=device)
    y = torch.tensor(class_labels, device=device)
    y_original = y.clone()

    # Setup classifier-free guidance:
    z = torch.cat([z, z], 0)
    y_null = torch.tensor([num_classes] * n, device=device)
    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y, cfg_scale=cfg_scale)

    # Sample images:
    samples = diffusion.p_sample_loop(
        model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    )
    #
    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    #samples = vae.decoder(samples / 0.18215, y_original)
    samples = vae.module.decoder(samples, y_original)
    # 
    # Save and display images:
    
    save_image(samples, f"./samples/{timestamp}-{rank}.png", nrow=10, normalize=True, value_range=(-1, 1))

def main():
    world_size = 4
    mp.spawn(sample, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    
    main()
