
import numpy as np
import pandas as pd
import onnxruntime as ort
import joblib
import time

# Load models
onnx_session = ort.InferenceSession(
    "battery_model.onnx",
    providers=["CPUExecutionProvider"]
)

sklearn_model = joblib.load("battery_model.pkl")

input_name = onnx_session.get_inputs()[0].name

# Generate 1000 simulated sensor readings
np.random.seed(42)

samples = np.column_stack([
    np.random.uniform(3.35, 4.20, 1000),
    np.random.uniform(8, 46, 1000),
    np.random.uniform(0, 37, 1000),
    np.random.uniform(-88, -45, 1000),
    np.random.uniform(0, 36, 1000)
]).astype(np.float32)

print("=" * 55)
print("SMART DOORBELL - OFFLINE EDGE AI BENCHMARK")
print("=" * 55)

# Warm-up
for sample in samples[:100]:
    onnx_session.run(
        None,
        {input_name: sample.reshape(1, 5)}
    )

# Measure individual ONNX inference
latencies = []
onnx_predictions = []

for sample in samples:
    input_data = sample.reshape(1, 5)

    start = time.perf_counter()

    result = onnx_session.run(
        None,
        {input_name: input_data}
    )[0][0][0]

    elapsed = (time.perf_counter() - start) * 1000

    latencies.append(elapsed)
    onnx_predictions.append(float(result))

latencies = np.array(latencies)
onnx_predictions = np.array(onnx_predictions)

# Calculate ONNX performance
average_latency = np.mean(latencies)
median_latency = np.median(latencies)
p95_latency = np.percentile(latencies, 95)
throughput = 1000 / average_latency

# Compare against original sklearn model
features = [
    "battery_v",
    "temp_c",
    "rings_per_day",
    "wifi_rssi_dbm",
    "age_months"
]

test_df = pd.DataFrame(samples, columns=features)

sklearn_predictions = sklearn_model.predict(test_df)
sklearn_predictions = np.maximum(sklearn_predictions, 0)

onnx_clipped = np.maximum(onnx_predictions, 0)

prediction_difference = np.mean(
    np.abs(sklearn_predictions - onnx_clipped)
)

# Display results
print("\nBENCHMARK RESULTS")
print("-" * 55)

print("Number of test samples:", len(samples))
print(f"Average latency: {average_latency:.5f} ms")
print(f"Median latency: {median_latency:.5f} ms")
print(f"P95 latency: {p95_latency:.5f} ms")
print(f"Throughput: {throughput:.2f} predictions/second")

print("\nMODEL CONSISTENCY")
print("-" * 55)

print(
    "Mean absolute prediction difference:",
    f"{prediction_difference:.6f} days"
)

print("\nRUNTIME INFORMATION")
print("-" * 55)

print("Inference engine: ONNX Runtime")
print("Execution provider: CPUExecutionProvider")
print("Model: battery_model.onnx")
print("External prediction API: Not used")

print("\nBenchmark completed successfully!")
print("=" * 55)
