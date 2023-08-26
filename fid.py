import os
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.datasets import LSUN
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import tqdm

import seaborn as sns
from torchvision.utils import make_grid
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from collections import OrderedDict
import torch.distributed as dist
from autoencoder import TAutoencoderKL, TVQVAE
from diffusers.models import AutoencoderKL
import imageio
from models import DiT_models
from diffusion import create_diffusion
#from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models

from scipy.linalg import sqrtm
from util import LSUNWithLabels

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 计算FID
def calculate_fid(act1, act2):
    mu1, sigma1 = act1.mean(dim=0), torch_cov(act1)
    mu2, sigma2 = act2.mean(dim=0), torch_cov(act2)
    ssdiff = torch.sum((mu1 - mu2)**2.0)
    covmean = sqrtm((sigma1.cpu().detach().numpy()).dot(sigma2.cpu().detach().numpy()))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + torch.trace(sigma1 + sigma2 - 2.0*torch.tensor(covmean).to(act1.device))
    return fid

# 计算协方差矩阵
def torch_cov(m, rowvar=False):
    if m.size(0) == 1:
        return torch.zeros(m.size(1), m.size(1)).to(m.device)
    if rowvar:
        m = m.t()
    fact = 1.0 / (m.size(1) - 1)
    m -= torch.mean(m, dim=1, keepdim=True)
    mt = m.t()
    return fact * m.matmul(mt).squeeze()

# 加载预训练的Inception v3模型
model = models.inception_v3(pretrained=True)
# 移除全连接层以得到2048维的特征向量
model.fc = nn.Identity()
model.eval()
model = model.cuda()  # 如果你使用GPU

# 计算Inception特征
def get_inception_features(images, model):
    with torch.no_grad():
        if images.device != next(model.parameters()).device:
    
            images = images.to(next(model.parameters()).device)
        features = model(images).squeeze(-1).squeeze(-1)
    return features

def sample(device, vae, num_samples):
    batch_size = 16
    model = "DiT-B/2"
    num_classes = 3
    num_samples = num_samples * num_classes
    cfg_scale = 4.0
    num_sampling_steps = 1000
    seed = torch.randint(0, 2 ** 32, (1,)).item()
    ckpt = '/workspace/project/Final_Project/results/010-DiT-B-2/checkpoints/0128920.pt'
    ckpt = "/workspace/project/Final_Project/results/011-DiT-B-2/checkpoints/0107824.pt"
    # Setup PyTorch:
    
    torch.manual_seed(seed)
    torch.set_grad_enabled(False)
    torch.cuda.set_device(device)

    
    # Load model:
    latent_size = 16
    latent_channels = 4
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
    sample_list = []
    while len(sample_list) < num_samples:

        # Labels to condition the model with (feel free to change):
        class_labels = []
        for i in range(num_classes):
            for _ in range(batch_size):
                class_labels.append(i)

        # Create sampling noise:
        n = len(class_labels)
        z = torch.randn(n, latent_channels, latent_size, latent_size, device=device)
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
        #samples = vae.decoder(samples, y_original)
        #samples = vae.decode(samples / 0.18215, return_dict=False)[0] 
        samples = vae.decoder(samples / 0.18215, y_original)
        # append all tensors in samples to a list
        
        for i in range(len(samples)):
            sample_list.append(samples[i])
        print("sample_list length: ", len(sample_list))
    # return a tensor of all samples
    return torch.stack(sample_list[:num_samples])

# 假设real_images和fake_images是你的实际和生成的图像，它们应该是FloatTensor类型，并缩放到[0,1]。
# real_images = ...
# fake_images = ...

# 确保图像的大小是(299, 299)，这是Inception v3模型所需要的。
# 如果不是，你需要调整它们的大小。
num_samples = 192
img_size = 299
data_path = "/workspace/project/data"
mean = (0.5, 0.5, 0.5)
std = (0.5, 0.5, 0.5)
transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

val_classes = ['bedroom_val', 'tower_val', 'bridge_val']
val_dataset = LSUNWithLabels(val_classes, root_dir=data_path, transform=transform, num_samples=num_samples)

# get all images as tensors from val_dataset    
real_images = []
for i in range(len(val_dataset)):
    real_images.append(val_dataset[i][0])
real_images = torch.stack(real_images)


dist.init_process_group('nccl', init_method='file:///tmp/dummyfile', rank=0, world_size=1)
latent_dim = (16,16,4)
ddp_path = "/workspace/project/TVQVAE/results/1693003607/TVQVAE.pt"
kl = torch.load(ddp_path, map_location=torch.device('cpu'))
kl = kl.module
kl = kl.to(device)
kl.eval()

fake_images = sample(device, kl, num_samples)
# resize fake images into (299, 299)
fake_images = F.interpolate(fake_images, size=(img_size, img_size), mode='bilinear', align_corners=False) 


# 计算特征
real_act = get_inception_features(real_images, model)
fake_act = get_inception_features(fake_images, model)

# 计算FID
fid_value = calculate_fid(real_act, fake_act)
print(f'FID score: {fid_value.item()}')
