from onnxruntime.quantization import quantize_dynamic, QuantType

# Apply dynamic quantization
quantized_model_path = "D:\prodigal-2\Pytorch_onnx\Self-supervised_Learning_Conversion\onnx\cat_dog_classifier_quantized.onnx"
quantize_dynamic("D:\prodigal-2\Pytorch_onnx\Self-supervised_Learning_Conversion\onnx/cat_dog_classifier.onnx", quantized_model_path, weight_type=QuantType.QUInt8)

print(f"Quantized model saved at {quantized_model_path}")
