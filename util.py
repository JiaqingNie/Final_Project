import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.datasets import LSUN

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