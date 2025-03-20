import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models, datasets
from torch.utils.data import DataLoader
from PIL import Image
import os
import glob

# Device Configuration
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Data Transformations for Training & Testing
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load Dataset
data_dir = "/Users/swedha/Documents/self-supervised_learning/PetImages"
dataset = datasets.ImageFolder(root=data_dir, transform=transform_train)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

