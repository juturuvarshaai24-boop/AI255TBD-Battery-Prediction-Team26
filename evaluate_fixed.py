
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("data/team_26_train.csv")

# Handle invalid sensor readings
df.loc[df["battery_v"] == 0, "battery_v"] = np.nan
df.loc[df["temp_c"] == -127, "temp_c"] = np.nan
df.loc[df["wifi_rssi_dbm"] == 0, "wifi_rssi_dbm"] = np.nan

X = df.drop("days_left", axis=1)
y = df["days_left"]

# Same test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Load trained model
model = joblib.load("battery_model.pkl")

# Predict
raw_predictions = model.predict(X_test)

# Apply non-negative constraint
fixed_predictions = np.maximum(raw_predictions, 0)

# Evaluate
print("BEFORE FIX")
print("MAE:", mean_absolute_error(y_test, raw_predictions))
print("RMSE:", np.sqrt(mean_squared_error(y_test, raw_predictions)))
print("R2:", r2_score(y_test, raw_predictions))

print("\nAFTER FIX")
print("MAE:", mean_absolute_error(y_test, fixed_predictions))
print("RMSE:", np.sqrt(mean_squared_error(y_test, fixed_predictions)))
print("R2:", r2_score(y_test, fixed_predictions))

print("\nNegative predictions after fix:",
      (fixed_predictions < 0).sum())

print("Minimum prediction:", fixed_predictions.min())

# Save fixed predictions for comparison
print("\nNon-negative prediction constraint applied.")
