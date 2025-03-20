import torch
import torch.nn as nn
import torchvision.models as models

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

# **Load Trained Model**
model = Classifier()
model.load_state_dict(torch.load("C:/Users/user/Desktop/Pytorch_onnx/Self-supervised_Learning_Technique(model)/model/cat_dog_classifier.pth", map_location=torch.device("cpu")))
model.eval()

# **Dummy Input for Export**
dummy_input = torch.randn(1, 3, 224, 224)  # Batch size 1, 3 color channels, 224x224 image

# **Export Model to ONNX**
torch.onnx.export(
    model, 
    dummy_input, 
    "C:/Users/user/Desktop/Pytorch_onnx/Self-supervised_Learning_Conversion/onnx/cat_dog_classifier.onnx", 
    input_names=["input"], 
    output_names=["output"], 
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=11
)

print("Model successfully converted to ONNX format and saved as 'cat_dog_classifier.onnx'")
