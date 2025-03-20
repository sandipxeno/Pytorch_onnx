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

# Function to Predict on New Dataset
def predict_from_dataset(dataset_path, model_path="cat_dog_classifier.pth"):
    # Load trained model
    model = Classifier().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Get image paths
    image_paths = glob.glob(os.path.join(dataset_path, "**", "*.jpg"), recursive=True) + \
                  glob.glob(os.path.join(dataset_path, "**", "*.png"), recursive=True)
    
    if not image_paths:
        print("No images found in the dataset folder.")
        return

    print(f"Found {len(image_paths)} images. Making predictions...")

    # Process and predict
    for image_path in image_paths[:5]:  # Predict first 5 images
        image = Image.open(image_path).convert("RGB")
        image = transform_test(image).unsqueeze(0).to(device)  # Add batch dimension

        with torch.no_grad():
            output = model(image)
            prediction = torch.argmax(output, dim=1).item()

        label = "Dog" if prediction == 1 else "Cat"
        print(f"Image: {image_path} -> Prediction: {label}")

# Predict on New Dataset
new_dataset_path = "/Users/swedha/Documents/data/Prediction dataset"  
predict_from_dataset(new_dataset_path)