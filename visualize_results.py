
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# Create results directory
os.makedirs("results", exist_ok=True)

# Load model comparison data
models = pd.read_csv("results/model_comparison.csv")

# GRAPH 1: MODEL ACCURACY COMPARISON (MAE)
plt.figure(figsize=(10, 6))

plt.bar(models["Model"], models["MAE (days)"])

plt.title("Model Accuracy Comparison - MAE")
plt.xlabel("Machine Learning Models")
plt.ylabel("Mean Absolute Error (Days)")
plt.xticks(rotation=20)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig("results/model_accuracy_comparison.png", dpi=300)
plt.show()


# GRAPH 2: ONNX FP32 VS INT8 COMPARISON
onnx_models = models[
    models["Model"].isin(["ONNX FP32", "ONNX INT8"])
]

plt.figure(figsize=(8, 5))

plt.bar(
    onnx_models["Model"],
    onnx_models["MAE (days)"]
)

plt.title("ONNX Model Quantization - MAE Comparison")
plt.xlabel("ONNX Model")
plt.ylabel("MAE (Days)")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig("results/onnx_quantization_comparison.png", dpi=300)
plt.show()


# GRAPH 3: INFERENCE LATENCY
latency_labels = ["Average", "Median", "P95"]
latency_values = [0.02999, 0.02732, 0.04102]

plt.figure(figsize=(8, 5))

plt.bar(latency_labels, latency_values)

plt.title("ONNX Runtime Inference Latency")
plt.xlabel("Latency Metric")
plt.ylabel("Latency (Milliseconds)")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig("results/inference_latency.png", dpi=300)
plt.show()


# GRAPH 4: EDGE AI PERFORMANCE SUMMARY
metrics = [
    "Average Latency (ms)",
    "P95 Latency (ms)",
    "Throughput (predictions/sec)",
    "Model Size (KB)"
]

values = [
    0.02999,
    0.04102,
    33342.23,
    4.00
]

summary = pd.DataFrame({
    "Metric": metrics,
    "Value": values
})

plt.figure(figsize=(10, 6))

plt.bar(summary["Metric"], summary["Value"])

plt.title("Smart Doorbell Edge AI - Performance Metrics")
plt.xlabel("Performance Metrics")
plt.ylabel("Measured Value")
plt.xticks(rotation=20)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig("results/edge_ai_performance_summary.png", dpi=300)
plt.show()


print("\nALL GRAPHS GENERATED SUCCESSFULLY!")
print("----------------------------------")

for file in os.listdir("results"):
    if file.endswith(".png"):
        print("Created:", file)
