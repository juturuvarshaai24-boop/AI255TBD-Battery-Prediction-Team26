
import pandas as pd

# Load original dataset
df = pd.read_csv("data/team_26_train.csv")

print("Original dataset shape:", df.shape)

# Replace suspicious sensor readings with missing values
df.loc[df["battery_v"] == 0, "battery_v"] = float("nan")

df.loc[df["temp_c"] == -127, "temp_c"] = float("nan")

df.loc[df["wifi_rssi_dbm"] == 0, "wifi_rssi_dbm"] = float("nan")

# Count missing values created
print("\nMissing values after anomaly handling:")
print(df.isnull().sum())

# Replace missing values using median
columns = ["battery_v", "temp_c", "wifi_rssi_dbm"]

for col in columns:
    df[col] = df[col].fillna(df[col].median())

# Verify
print("\nMissing values after imputation:")
print(df.isnull().sum())

print("\nCleaned dataset shape:", df.shape)

# Save cleaned dataset
df.to_csv("data/team_26_clean.csv", index=False)

print("\nCleaned dataset saved successfully!")
