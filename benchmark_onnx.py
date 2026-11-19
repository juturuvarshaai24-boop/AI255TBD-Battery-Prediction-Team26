
import time
import numpy as np
import pandas as pd
import joblib
import onnxruntime as ort

# Load original model
original_model = joblib.load("battery_model.pkl")

# Load ONNX model
session = ort.InferenceSession(
    "battery_model.onnx",
    providers=["CPUExecutionProvider"]
)

# Sample sensor reading
sample = pd.DataFrame([{
    "battery_v": 3.8,
    "temp_c": 28,
    "rings_per_day": 8,
    "wifi_rssi_dbm": -65,
    "age_months": 12
}])

onnx_input = sample.to_numpy(dtype=np.float32)
input_name = session.get_inputs()[0].name

# Warm-up both models
for _ in range(100):
    original_model.predict(sample)
    session.run(None, {input_name: onnx_input})

# Benchmark original model
original_times = []

for _ in range(1000):
    start = time.perf_counter()
    original_model.predict(sample)
    end = time.perf_counter()
    original_times.append((end - start) * 1000)

# Benchmark ONNX model
onnx_times = []

for _ in range(1000):
    start = time.perf_counter()
    session.run(None, {input_name: onnx_input})
    end = time.perf_counter()
    onnx_times.append((end - start) * 1000)

# Display results
def show_results(name, times):
    print(f"\n{name}")
    print("-" * 30)
    print("Average latency:", round(np.mean(times), 5), "ms")
    print("Median latency:", round(np.percentile(times, 50), 5), "ms")
    print("P95 latency:", round(np.percentile(times, 95), 5), "ms")
    print("Predictions per second:",
          round(1000 / np.mean(times), 2))


print("EDGE AI INFERENCE BENCHMARK")
print("===========================")

show_results("Original Scikit-learn Model", original_times)
show_results("ONNX Runtime Model", onnx_times)

speedup = np.mean(original_times) / np.mean(onnx_times)

print("\nSPEED COMPARISON")
print("----------------")
print("ONNX speedup:", round(speedup, 3), "x")
