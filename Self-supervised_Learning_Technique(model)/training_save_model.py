import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models, datasets
from torch.utils.data import DataLoader
from PIL import Image
import os
import glob
from self_supervised import models,device, dataloader, transform_test
from supervised_learning_model_and_loss_function import model, optimizer, criterion, Classifier

# Training Function
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for (x, y) in dataloader:
        x, y = x.to(device), y.to(device)

        outputs = model(x)
        loss = criterion(outputs, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

# Train Model
for epoch in range(10):
    loss = train(model, dataloader, optimizer, criterion, device)
    print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

# Save the Model
model_path = "cat_dog_classifier.pth"
torch.save(model.state_dict(), model_path)
print(f"Model saved at {model_path}")
