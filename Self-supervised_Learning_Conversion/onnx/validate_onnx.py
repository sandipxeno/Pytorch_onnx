import onnx

# Load ONNX model
onnx_model = onnx.load("D:\prodigal-2\Pytorch_onnx\Self-supervised_Learning_Conversion\onnx/cat_dog_classifier.onnx")

# Validate model
onnx.checker.check_model(onnx_model)
print("ONNX model is valid")
