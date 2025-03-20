import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models, datasets
from torch.utils.data import DataLoader
from PIL import Image
import os
import glob

# **Device Configuration**
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# **Data Transformations for Training & Testing**
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# **Load Dataset**
data_dir = "C:\Users\user\Desktop\Pytorch_onnx\Dataset\train"
dataset = datasets.ImageFolder(root=data_dir, transform=transform_train)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# **Modified Encoder with Classification Head**
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

# **Load the Pretrained Encoder**
model = Classifier().to(device)

# **Loss Function & Optimizer**
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

# **Training Function**
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

# **Train Model**
for epoch in range(10):
    loss = train(model, dataloader, optimizer, criterion, device)
    print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

# **Save the Model**
model_path = "cat_dog_classifier.pth"
torch.save(model.state_dict(), model_path)
print(f"Model saved at {model_path}")

# **Function to Predict on New Dataset**
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

# **Example Usage: Predict on New Dataset**
new_dataset_path = "/Users/swedha/Documents/data/Prediction dataset"  # Change path
predict_from_dataset(new_dataset_path)
