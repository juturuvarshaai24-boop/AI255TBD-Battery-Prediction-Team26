
import pandas as pd

df = pd.read_csv("data/team_26_train.csv")

print("BATTERY VOLTAGE ZERO:")
print(df[df["battery_v"] == 0])

print("\nTEMPERATURE -127:")
print(df[df["temp_c"] == -127])

print("\nWIFI RSSI ZERO:")
print(df[df["wifi_rssi_dbm"] == 0])

print("\nRINGS PER DAY ABOVE 20:")
print(df[df["rings_per_day"] > 20])

print("\nTOTAL SUSPICIOUS VALUES:")
print("Battery voltage zero:", (df["battery_v"] == 0).sum())
print("Temperature -127:", (df["temp_c"] == -127).sum())
print("WiFi RSSI zero:", (df["wifi_rssi_dbm"] == 0).sum())
