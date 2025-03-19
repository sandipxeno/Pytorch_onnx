from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os

# **Initialize Flask App**
app = Flask(__name__)

# **Device Configuration**
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# **Define Transform for Input Image**
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# **Define Classifier Model**
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

# **Load Model**
MODEL_PATH = "cat_dog_classifier.pth"

def load_model():
    model = Classifier().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model

model = load_model()

# **Define Prediction Function**
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform_test(image).unsqueeze(0).to(device)  # Add batch dimension

    with torch.no_grad():
        output = model(image)
        prediction = torch.argmax(output, dim=1).item()

    return "Dog" if prediction == 1 else "Cat"

# **Flask Routes**
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Save uploaded file
    file_path = os.path.join("uploads", file.filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(file_path)

    # Predict
    result = predict_image(file_path)

    return jsonify({"prediction": result})

# **Run Flask App**
if __name__ == "__main__":
    app.run(debug=True)
