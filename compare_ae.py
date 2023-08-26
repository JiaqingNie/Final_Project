import os
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
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

def denorm(img_tensors, mean, std):
    # denormalize image tensors with mean and std of training dataset for all channels
    img_tensors = img_tensors.permute(1, 0, 2, 3)
    for t, m, s in zip(img_tensors, mean, std):
        t.mul_(s).add_(m)
    img_tensors = img_tensors.permute(1, 0, 2, 3)
    return img_tensors
    
    
def save_image(img, path, name):
    npimg = img.cpu().numpy()
    npimg = np.transpose(npimg, (1,2,0))
    npimg = (npimg * 255).astype(np.uint8)
    # save image
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    imageio.imwrite(f'{path}/{name}.png', npimg)

dist.init_process_group('nccl', init_method='file:///tmp/dummyfile', rank=0, world_size=1)

ddp_path = "/workspace/project/TVQVAE/results/1691516614/TVQVAE.pt"
state_dict = torch.load(ddp_path, map_location=torch.device('cpu')).state_dict()
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] if k.startswith('module.') else k  # 删除'module.'前缀
    new_state_dict[name] = v

batch_size = 4
img_size = 64
n_embed = 2048
num_samples = 500
ckpt_period = 2
disc_train_period = 1
disc_start = 37501
data_path = "/workspace/project/data"


#vq.load_state_dict(new_state_dict)
vq = torch.load(ddp_path, map_location=torch.device('cpu'))
# change DDP to non-DDP
vq = vq.module
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
vq = vq.to(device)
vq.eval()

latent_dim = (16,16,4)
ddp_path = "/workspace/project/T-Autoencoder-KL/results/1692483020/TAKL-9.pt"
state_dict = torch.load(ddp_path, map_location=torch.device('cpu')).state_dict()
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] if k.startswith('module.') else k  # 删除'module.'前缀
    new_state_dict[name] = v
kl = TAutoencoderKL(latent_dim=latent_dim, image_size=img_size, patch_size=2, in_channels=3, hidden_size=768, depth=12, num_heads=6, mlp_ratio=6.0, num_classes=3, dropout_prob=0.1, disc_start=disc_start,
                               kl_weight=1.0e-6,
                               num_epochs=8)
kl.load_state_dict(new_state_dict)
kl = kl.to(device)
kl.eval()

ddp_path = "/workspace/project/Autoencoder-KL/results/1692965183/TAKL-3.pt"
ae = torch.load(ddp_path, map_location=torch.device('cpu'))
# change DDP to non-DDP
ae = ae.module
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
ae = ae.to(device)
ae.eval()

mean = (0.5, 0.5, 0.5)
std = (0.5, 0.5, 0.5)
transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

val_classes = ['bedroom_val', 'tower_val', 'bridge_val']
val_dataset = LSUNWithLabels(val_classes, root_dir=data_path, transform=transform, num_samples=500)
val_loader = DataLoader(val_dataset, batch_size, shuffle=True)


rec_losses1 = {}
rec_losses2 = {}
rec_losses3 = {}
for i in range(3):
    rec_losses1[i] = []
    rec_losses2[i] = []
    rec_losses3[i] = []
with torch.no_grad():
    # randomly choose 10 images from test set and for each image, generate 10 samples, save them and the original image in a grid
        # get 10 images from loader_test
    for j, (x, y) in enumerate(val_loader):
        x = x.to(device)
        y = y.to(device)
        rec1, _, _ = vq(x, y)
        rec2, _ = kl(x, y)
        rec3 = ae(x, return_dict=False)[0]
        
        # calculate reconstruction loss for each image between x and recons
        losses1 = F.mse_loss(x, rec1, reduction='none')
        # keep the first dim, average the rest
        losses1 = losses1.view(losses1.shape[0], -1).mean(-1)
        
        losses2 = F.mse_loss(x, rec2, reduction='none')
        losses2 = losses2.view(losses2.shape[0], -1).mean(-1)
        
        losses3 = F.mse_loss(x, rec3, reduction='none')
        losses3 = losses3.view(losses3.shape[0], -1).mean(-1)
       
        for i in range(len(losses1)):
            rec_losses1[y[i].item()].append(losses1[i].item())
        
        for i in range(len(losses2)):
            rec_losses2[y[i].item()].append(losses2[i].item())
        
        for i in range(len(losses3)):
            rec_losses3[y[i].item()].append(losses3[i].item())
        
        if j < 10:
            rec1 = rec1.cpu()
            rec1 = make_grid(torch.clamp(denorm(rec1, mean, std), 0., 1.), nrow=2, padding=2, normalize=False,
                                    range=None, scale_each=False, pad_value=1)
            
            # remove white padding around images
            
            rec2 = rec2.cpu()
            rec2 = make_grid(torch.clamp(denorm(rec2, mean, std), 0., 1.), nrow=2, padding=2, normalize=False,
                                    range=None, scale_each=False, pad_value=1)
            rec3 = rec3.cpu()
            rec3 = make_grid(torch.clamp(denorm(rec3, mean, std), 0., 1.), nrow=2, padding=2, normalize=False,
                                    range=None, scale_each=False, pad_value=1)
            
            xs = x.cpu()
            xs = make_grid(torch.clamp(denorm(xs, mean, std), 0., 1.), nrow=2, padding=2, normalize=False,
                                    range=None, scale_each=False, pad_value=1)
            #plt.figure(figsize = (8,8))
            save_image(rec1, f"/workspace/project/eval", f"rec_vq_{j}")
            
            #plt.figure(figsize = (8,8))
            save_image(rec2, f"/workspace/project/eval", f"rec_kl_{j}")
            
            #plt.figure(figsize = (8,8))
            save_image(rec3, f"/workspace/project/eval", f"rec_ae_{j}")
            
            #plt.figure(figsize = (8,8))
            save_image(xs, f"/workspace/project/eval", f"xs_{j}")

# plot the distribution of reconstruction losses for each class within the same plot
classes = ('plane', 'car', 'bird', 'cat',
        'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
plt.figure(figsize = (8,8))
for i in range(3):
    sns.distplot(rec_losses1[i], hist=False, label=classes[i])
    sns.distplot(rec_losses2[i], hist=False, label=classes[i])
    sns.distplot(rec_losses3[i], hist=False, label=classes[i])
plt.legend(loc='upper right')
plt.xlabel('Reconstruction Loss')
plt.ylabel('Density')
plt.savefig(f"/workspace/project/eval/rec_loss.png")
plt.close()

# calculate the average reconstruction loss for each class
avg_rec_losses1 = {}
for i in range(3):
    avg_rec_losses1[i] = np.mean(rec_losses1[i])
print(avg_rec_losses1)


# calculate the average reconstruction loss for the whole test set
avg_rec_loss1 = np.mean([avg_rec_losses1[i] for i in range(3)])
print(f"The reconstruction loss of VQ for the while test set: {avg_rec_loss1}.")


avg_rec_losses2 = {}
for i in range(3):
    avg_rec_losses2[i] = np.mean(rec_losses2[i])
print(avg_rec_losses2)

avg_rec_loss2 = np.mean([avg_rec_losses2[i] for i in range(3)])
print(f"The reconstruction loss of KL for the while test set: {avg_rec_loss2}.")

avg_rec_losses3 = {}
for i in range(3):
    avg_rec_losses3[i] = np.mean(rec_losses3[i])
print(avg_rec_losses3)

avg_rec_loss3 = np.mean([avg_rec_losses3[i] for i in range(3)])
print(f"The reconstruction loss of AE for the while test set: {avg_rec_loss3}.")
