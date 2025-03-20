import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models, datasets
from torch.utils.data import DataLoader
from PIL import Image
import os
import glob
from self_supervised import models 
from self_supervised import device

# Modified Encoder with Classification Head
class Classifier(nn.Module):
    def __init__(self, feature_dim=128, num_classes=2):
        super(Classifier, self).__init__()
        self.encoder = models.resnet18(weights=None)  # No pretrained weights
        self.encoder.fc = nn.Linear(512, feature_dim)  # Feature Extractor
        self.classifier = nn.Linear(feature_dim, num_classes)  # Classification Head

    def forward(self, x):
        features = self.encoder(x)
        output = self.classifier(features)
        return output

# Load the Pretrained Encoder
model = Classifier().to(device)

# Loss Function & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)