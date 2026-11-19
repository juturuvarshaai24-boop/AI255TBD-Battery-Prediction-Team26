
import os
import onnx
from onnxruntime.quantization import (
    quantize_dynamic,
    QuantType
)

# Original ONNX model
input_model = "battery_model.onnx"

# Quantised ONNX model
output_model = "battery_model_int8.onnx"

# Apply dynamic INT8 quantisation
quantize_dynamic(
    model_input=input_model,
    model_output=output_model,
    weight_type=QuantType.QInt8
)

# Verify the quantised model
model = onnx.load(output_model)
onnx.checker.check_model(model)

# Compare model sizes
original_size = os.path.getsize(input_model) / 1024
quantized_size = os.path.getsize(output_model) / 1024

print("INT8 QUANTISATION SUCCESSFUL!")
print("--------------------------------")

print("Original ONNX size:",
      round(original_size, 2), "KB")

print("INT8 ONNX size:",
      round(quantized_size, 2), "KB")

reduction = (
    (original_size - quantized_size)
    / original_size
) * 100

print("Size reduction:",
      round(reduction, 2), "%")

