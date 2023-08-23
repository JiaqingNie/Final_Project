# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torchvision.datasets import LSUN
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torchvision
import numpy as np
import torch.multiprocessing as mp
from collections import OrderedDict
from diffusion import create_diffusion
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
from torchvision.utils import save_image

from models import DiT_models
from diffusion import create_diffusion
from autoencoder import TVQVAE

def eval(model, vae, rank, epoch=0,num_classes=3, cfg_scale=4.0, latent_size=16, num_sampling_steps=1000, channels=4):
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)
    
    class_labels = []
    for i in range(num_classes):
        for _ in range(5):
            class_labels.append(i)

    # Create sampling noise:
    n = len(class_labels)
    z = torch.randn(n, channels, latent_size, latent_size, device=device)
    y = torch.tensor(class_labels, device=device)
    y_original = y.clone()

    # Setup classifier-free guidance:
    z = torch.cat([z, z], 0)
    y_null = torch.tensor([num_classes] * n, device=device)
    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y, cfg_scale=cfg_scale)
    diffusion = create_diffusion(str(num_sampling_steps))
    # Sample images:
    samples = diffusion.p_sample_loop(
        model.module.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    )
    #
    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    samples = vae.module.decoder(samples / 0.18215, y_original)
    # 
    # Save and display images:
    
    save_image(samples, f"./eval/{epoch}.png", nrow=5, normalize=True, value_range=(-1, 1))

class LSUNWithLabels:
    def __init__(self, lsun_classes, root_dir, transform=None, num_samples=50000):
        self.datasets = []
        
        for idx, cls in enumerate(lsun_classes):
            full_dataset = LSUN(root=root_dir, classes=[cls], transform=transform)
            
            def wrapped_getitem(index, dataset=full_dataset):
                data, _ = dataset[index]
                return data
            
            indices = torch.randperm(len(full_dataset))[:num_samples]
            subset = [(wrapped_getitem(index), idx) for index in indices]
            self.datasets.append(subset)
            
        self.lengths = [len(subset) for subset in self.datasets]
        self.total_length = sum(self.lengths)

    def __getitem__(self, index):
        for i, subset in enumerate(self.datasets):
            if index < self.lengths[i]:
                return subset[index]
            index -= self.lengths[i]
        raise IndexError

    def __len__(self):
        return self.total_length


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12357'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


#################################################################################
#                                  Training Loop                                #
#################################################################################

def train(rank, world_size):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    setup(rank, world_size)
    latent_size = 16
    num_samples = 50000
    latent_channels = 3
    data_path = "/project/TVQVAE/data/lsun"
    vae_path = "/project/TVQVAE/results/1691516614/TVQVAE.pt"
    results_dir = "./results"
    batch_size = 16
    num_classes = 3
    img_size = 64
    epochs = 100
    global_seed = 0
    model = "DiT-B/2"
    num_workers = 4
    log_every = 100
    ckpt_every = 1
    
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{results_dir}/*"))
        model_string_name = model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        experiment_dir = f"{results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    
    model = DiT_models[model](
        input_size=latent_size,
        num_classes=num_classes,
        in_channels=latent_channels,
    )
    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    model = DDP(model.to(device), device_ids=[rank])
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    vae = torch.load(vae_path)
    vae = vae.to(device)
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)


    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


    lsun_classes = ['bedroom_train', 'tower_train', 'bridge_train']
    dataset = LSUNWithLabels(lsun_classes, root_dir=data_path, transform=transform, num_samples=num_samples)    
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler
    )
    
    logger.info(f"Dataset contains {len(dataset):,} images ({data_path})")

    # Prepare models for training:
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode
    vae.eval()

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()
    total_iters = len(loader)
    
    args = {
        "latent_size": latent_size,
        "num_samples": num_samples,
        "data_path": data_path,
        "vae_path": vae_path,
        "results_dir": results_dir,
        "batch_size": batch_size,
        "num_classes": num_classes,
        "img_size": img_size,
        "epochs": epochs,
        "global_seed": global_seed,
        "model": model
    }
    if rank == 0:
        checkpoint = {
            "model": model.module.state_dict(),
            "ema": ema.state_dict(),
            "opt": opt.state_dict(),
            "args": args
        }
        checkpoint_path = f"{checkpoint_dir}/{train_steps//ckpt_every:07d}.pt"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        eval(model, vae, rank=rank, epoch=0, latent_size=latent_size, channels=latent_channels)
    dist.barrier()
    
    logger.info(f"Training for {epochs} epochs...")
    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for idx, (x, y) in  enumerate(loader):
            x = x.to(device)
            y = y.to(device)
            
            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                x = vae.module.encoder(x, y)
                x, _ = vae.module.codebook.straight_through(x)
                x = x.mul(0.18215)
            
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(y=y)
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model.module)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(idx={idx}/{total_iters} step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            
            # Save DiT checkpoint:
        if epoch % ckpt_every == 0 or epoch == epochs - 1:
            if rank == 0:
                checkpoint = {
                    "model": model.module.state_dict(),
                    "ema": ema.state_dict(),
                    "opt": opt.state_dict(),
                    "args": args
                }
                checkpoint_path = f"{checkpoint_dir}/{train_steps//ckpt_every:07d}.pt"
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
                eval(model, vae, rank=rank, epoch=epoch+1, latent_size=latent_size, channels=latent_channels)
            dist.barrier()

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()

def main():
    world_size = 4
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
