
import numpy as np
import onnxruntime as ort

# Test sensor reading
sample = np.array(
    [[3.8, 28, 8, -65, 12]],
    dtype=np.float32
)

# Load original ONNX model
fp32_session = ort.InferenceSession(
    "battery_model.onnx",
    providers=["CPUExecutionProvider"]
)

# Load INT8 model
int8_session = ort.InferenceSession(
    "battery_model_int8.onnx",
    providers=["CPUExecutionProvider"]
)

# Get input names
fp32_input = fp32_session.get_inputs()[0].name
int8_input = int8_session.get_inputs()[0].name

# Generate predictions
fp32_prediction = fp32_session.run(
    None, {fp32_input: sample}
)[0][0][0]

int8_prediction = int8_session.run(
    None, {int8_input: sample}
)[0][0][0]

# Compare results
difference = abs(
    float(fp32_prediction) - float(int8_prediction)
)

print("FP32 vs INT8 PREDICTION")
print("-----------------------")
print("FP32 prediction:", fp32_prediction)
print("INT8 prediction:", int8_prediction)
print("Absolute difference:", difference)
print("Difference in days:", difference)
