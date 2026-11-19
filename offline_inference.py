
import numpy as np
import onnxruntime as ort
import time
import os

# Load the trained ONNX model
model_path = "battery_model.onnx"

session = ort.InferenceSession(
    model_path,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name

print("=" * 45)
print("SMART DOORBELL - OFFLINE EDGE AI")
print("=" * 45)

# Enter sensor readings
battery_v = float(input("Battery Voltage (V): "))
temp_c = float(input("Temperature (C): "))
rings = float(input("Rings Per Day: "))
wifi = float(input("WiFi Signal (dBm): "))
age = float(input("Battery Age (Months): "))

# Validate inputs against observed dataset ranges
if not 3.35 <= battery_v <= 4.20:
    raise ValueError("Battery voltage must be between 3.35 and 4.20 V.")

if not 8 <= temp_c <= 46:
    raise ValueError("Temperature must be between 8 and 46 C.")

if not 0 <= rings <= 37:
    raise ValueError("Rings per day must be between 0 and 37.")

if not -88 <= wifi <= -45:
    raise ValueError("WiFi signal must be between -88 and -45 dBm.")

if not 0 <= age <= 36:
    raise ValueError("Battery age must be between 0 and 36 months.")

# Prepare model input
input_data = np.array(
    [[battery_v, temp_c, rings, wifi, age]],
    dtype=np.float32
)

# Run local ONNX inference
start = time.perf_counter()

prediction = session.run(
    None,
    {input_name: input_data}
)[0][0][0]

latency = (time.perf_counter() - start) * 1000

# Display result
print("\n" + "=" * 45)
print("OFFLINE PREDICTION RESULT")
print("=" * 45)

print(f"Estimated Battery Life: {max(float(prediction), 0):.2f} days")
print(f"Inference Latency: {latency:.5f} ms")
print(f"Model Size: {os.path.getsize(model_path) / 1024:.2f} KB")
print("Inference Engine: ONNX Runtime")
print("Execution Provider: CPUExecutionProvider")
print("External Prediction API: Not Used")
print("=" * 45)
