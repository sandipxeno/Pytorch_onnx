import onnx

# Load ONNX model
onnx_model = onnx.load("D:/prodigal-2/Pytorch_onnx/Data_Labelling_Conversion/onnx/resnet50_dog_cat.onnx")

# Validate model
onnx.checker.check_model(onnx_model)
print("ONNX model is valid")
