
import joblib
import pandas as pd
import numpy as np

# Load trained model
model = joblib.load("battery_model.pkl")

# Observed valid ranges from the dataset
VALID_RANGES = {
    "battery_v": (3.35, 4.20),
    "temp_c": (8, 46),
    "rings_per_day": (0, 37),
    "wifi_rssi_dbm": (-88, -45),
    "age_months": (0, 36)
}


def predict_battery_life(battery_v, temp_c, rings_per_day,
                         wifi_rssi_dbm, age_months):

    readings = {
        "battery_v": battery_v,
        "temp_c": temp_c,
        "rings_per_day": rings_per_day,
        "wifi_rssi_dbm": wifi_rssi_dbm,
        "age_months": age_months
    }

    # Validate sensor readings
    for feature, value in readings.items():
        minimum, maximum = VALID_RANGES[feature]

        if not minimum <= value <= maximum:
            raise ValueError(
                f"{feature} must be between {minimum} and {maximum}."
            )

    # Prepare input in the same order as training
    input_data = pd.DataFrame([readings])

    # Predict battery life
    prediction = model.predict(input_data)

    # Ensure non-negative output
    return max(float(prediction[0]), 0)


print("SMART DOORBELL BATTERY LIFE PREDICTION")
print("--------------------------------------")

try:
    battery_v = float(input("Battery voltage (V): "))
    temp_c = float(input("Temperature (C): "))
    rings_per_day = int(input("Rings per day: "))
    wifi_rssi_dbm = int(input("WiFi signal strength (dBm): "))
    age_months = int(input("Battery age (months): "))

    days = predict_battery_life(
        battery_v, temp_c, rings_per_day,
        wifi_rssi_dbm, age_months
    )

    print("\nPREDICTION RESULT")
    print("-----------------")
    print("Estimated remaining battery life:",
          round(days, 2), "days")

except ValueError as error:
    print("\nINVALID INPUT:", error)
