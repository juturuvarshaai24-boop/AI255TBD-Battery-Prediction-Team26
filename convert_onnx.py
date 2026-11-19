
import joblib
import numpy as np
import pandas as pd
import os
import onnx

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort

# 1. Load the existing trained model
model = joblib.load("battery_model.pkl")

# 2. Define the five input features
initial_type = [
    ("float_input", FloatTensorType([None, 5]))
]

# 3. Convert the model to ONNX
onnx_model = convert_sklearn(
    model,
    initial_types=initial_type,
    target_opset=12
)

# 4. Save the ONNX model
onnx.save_model(onnx_model, "battery_model.onnx")

# 5. Verify the ONNX file
onnx.checker.check_model(onnx_model)

print("ONNX conversion successful!")

# 6. Load the ONNX model
session = ort.InferenceSession(
    "battery_model.onnx",
    providers=["CPUExecutionProvider"]
)

# 7. Prepare sample sensor readings
sample = pd.DataFrame([{
    "battery_v": 3.8,
    "temp_c": 28,
    "rings_per_day": 8,
    "wifi_rssi_dbm": -65,
    "age_months": 12
}])

# 8. Predict using original model
original_prediction = model.predict(sample)[0]

# 9. Predict using ONNX Runtime
input_name = session.get_inputs()[0].name
onnx_input = sample.to_numpy(dtype=np.float32)

onnx_prediction = session.run(
    None,
    {input_name: onnx_input}
)[0][0][0]

# 10. Compare predictions
print("\nMODEL COMPARISON")
print("----------------")
print("Original model:", round(float(original_prediction), 4))
print("ONNX model:", round(float(onnx_prediction), 4))
print(
    "Absolute difference:",
    round(abs(float(original_prediction) - float(onnx_prediction)), 8)
)

# 11. Compare model file sizes
original_size = os.path.getsize("battery_model.pkl") / 1024
onnx_size = os.path.getsize("battery_model.onnx") / 1024

print("\nMODEL SIZE COMPARISON")
print("---------------------")
print("Original model:", round(original_size, 2), "KB")
print("ONNX model:", round(onnx_size, 2), "KB")
