from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os
from werkzeug.utils import secure_filename
from collections import OrderedDict

# Initialize Flask App
app = Flask(__name__)

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Allowed File Extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Define Image Transform (Updated)
transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Define Model Class
class Classifier(nn.Module):
    def __init__(self, num_classes=2):
        super(Classifier, self).__init__()
        self.encoder = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.encoder.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        return self.encoder(x)

# Load Model

MODEL_PATH =os.path.join(os.getcwd(), "model", "resnet50_dog_cat.pth")
def load_model():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{MODEL_PATH}' not found!")
        return None
    
    model = Classifier().to(device)
    state_dict = torch.load(MODEL_PATH, map_location=device)

    # Fix if saved using DataParallel (multi-GPU)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace("module.", "")  # Remove "module." if present
        new_state_dict[new_key] = v

    try:
        model.load_state_dict(new_state_dict,strict=False)
    except RuntimeError as e:
        print("❌ Model loading error! Possible size mismatch.")
        print(e)
        return None

    model.eval()
    print("✅ Model loaded successfully.")
    return model

model = load_model()

# Define Prediction Function
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform_test(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        prediction = torch.argmax(output, dim=1).item()

    return "Dog" if prediction == 1 else "Cat"

# Flask Routes
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

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    # Save uploaded file
    filename = secure_filename(file.filename)
    file_path = os.path.join("uploads", filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(file_path)

    # Predict
    result = predict_image(file_path)

    return jsonify({"prediction": result})

# Run Flask App
if __name__ == "__main__":
    app.run(debug=True)



