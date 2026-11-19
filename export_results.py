
import pandas as pd
import json
import os

# Create results folder
os.makedirs("results", exist_ok=True)

# 1. OFFLINE BENCHMARK RESULTS
benchmark = pd.DataFrame({
    "Metric": [
        "Number of test samples",
        "Average latency (ms)",
        "Median latency (ms)",
        "P95 latency (ms)",
        "Throughput (predictions/sec)",
        "Mean absolute prediction difference (days)"
    ],
    "Value": [
        1000,
        0.02999,
        0.02732,
        0.04102,
        33342.23,
        0.000004
    ]
})

benchmark.to_csv(
    "results/offline_benchmark_results.csv",
    index=False
)

# 2. MODEL COMPARISON
models = pd.DataFrame({
    "Model": [
        "Dummy Regressor",
        "Linear Regression",
        "Neural Network",
        "ONNX FP32",
        "ONNX INT8"
    ],
    "MAE (days)": [
        51.297,
        22.629,
        6.669,
        6.669,
        6.716
    ],
    "RMSE (days)": [
        None,
        None,
        11.005,
        11.005,
        10.995
    ],
    "R2 Score": [
        None,
        None,
        0.968864,
        0.968864,
        0.968919
    ]
})

models.to_csv(
    "results/model_comparison.csv",
    index=False
)

# 3. PERFORMANCE SUMMARY
summary = {
    "project": "Smart Doorbell Battery-Life Prediction",
    "model": "ONNX FP32",
    "runtime": "ONNX Runtime",
    "execution_provider": "CPUExecutionProvider",
    "external_prediction_api": False,
    "test_samples": 1000,
    "average_latency_ms": 0.02999,
    "throughput_predictions_per_second": 33342.23,
    "model_size_kb": 4.00,
    "mean_prediction_difference_days": 0.000004
}

with open(
    "results/performance_summary.json",
    "w"
) as file:
    json.dump(summary, file, indent=4)

print("EXPERIMENT RESULTS EXPORTED SUCCESSFULLY!")
print("-----------------------------------------")
print("1. results/offline_benchmark_results.csv")
print("2. results/model_comparison.csv")
print("3. results/performance_summary.json")

