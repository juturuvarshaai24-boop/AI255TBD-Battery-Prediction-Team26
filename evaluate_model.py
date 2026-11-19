
import pandas as pd
import numpy as np
import joblib
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Load original dataset
df = pd.read_csv("data/team_26_train.csv")

# Handle invalid sensor readings
df.loc[df["battery_v"] == 0, "battery_v"] = np.nan
df.loc[df["temp_c"] == -127, "temp_c"] = np.nan
df.loc[df["wifi_rssi_dbm"] == 0, "wifi_rssi_dbm"] = np.nan

X = df.drop("days_left", axis=1)
y = df["days_left"]

# Recreate the same test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Load saved model
model = joblib.load("battery_model.pkl")

# Measure prediction time
start = time.perf_counter()
predictions = model.predict(X_test)
elapsed = time.perf_counter() - start

# Evaluate
print("MODEL VERIFICATION")
print("MAE:", mean_absolute_error(y_test, predictions))
print("R2:", r2_score(y_test, predictions))

print("\nPREDICTION CHECK")
print("Minimum prediction:", predictions.min())
print("Maximum prediction:", predictions.max())
print("Negative predictions:", (predictions < 0).sum())

print("\nINFERENCE PERFORMANCE")
print("Total prediction time:", elapsed, "seconds")
print("Average time per sample:", elapsed / len(X_test), "seconds")

print("\nSAMPLE PREDICTIONS")
for actual, predicted in zip(y_test.iloc[:10], predictions[:10]):
    print(
        f"Actual: {actual:.2f} days | "
        f"Predicted: {predicted:.2f} days"
    )
