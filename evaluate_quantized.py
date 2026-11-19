
import numpy as np
import pandas as pd
import onnxruntime as ort
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load original dataset
df = pd.read_csv("data/team_26_train.csv")

# Handle invalid sensor readings exactly as before
df["battery_v"] = df["battery_v"].replace(0, np.nan)
df["temp_c"] = df["temp_c"].replace(-127, np.nan)
df["wifi_rssi_dbm"] = df["wifi_rssi_dbm"].replace(0, np.nan)

X = df.drop(columns=["days_left"])
y = df["days_left"]

# Recreate the original test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Load original model
original_model = joblib.load("battery_model.pkl")

# Load ONNX models
fp32_session = ort.InferenceSession("battery_model.onnx")
int8_session = ort.InferenceSession("battery_model_int8.onnx")

fp32_input = fp32_session.get_inputs()[0].name
int8_input = int8_session.get_inputs()[0].name

# Original model predictions
fp32_predictions = original_model.predict(X_test)
fp32_predictions = np.maximum(fp32_predictions, 0)

# ONNX input: preserve the five feature columns and order
test_input = X_test.to_numpy(dtype=np.float32)

# ONNX FP32 predictions
fp32_onnx_predictions = fp32_session.run(
    None, {fp32_input: test_input}
)[0].reshape(-1)

# ONNX INT8 predictions
int8_predictions = int8_session.run(
    None, {int8_input: test_input}
)[0].reshape(-1)

# Apply the same non-negative output rule
fp32_onnx_predictions = np.maximum(fp32_onnx_predictions, 0)
int8_predictions = np.maximum(int8_predictions, 0)

# Calculate metrics
def evaluate(name, predictions):
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print(f"\n{name}")
    print("---------------------------")
    print("MAE:", round(mae, 4))
    print("RMSE:", round(rmse, 4))
    print("R2:", round(r2, 6))

print("FULL TEST DATASET COMPARISON")
print("============================")
print("Test samples:", len(X_test))

evaluate("Original Scikit-learn", fp32_predictions)
evaluate("ONNX FP32", fp32_onnx_predictions)
evaluate("ONNX INT8", int8_predictions)

print("\nINT8 vs FP32 ONNX")
print("-----------------")
print(
    "Mean absolute prediction difference:",
    round(float(np.mean(np.abs(
        fp32_onnx_predictions - int8_predictions
    ))), 6)
)
