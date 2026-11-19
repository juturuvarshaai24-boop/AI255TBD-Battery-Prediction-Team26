
import pandas as pd
import matplotlib.pyplot as plt

# Load Team 26 dataset
df = pd.read_csv("data/team_26_train.csv")

print("Dataset shape:", df.shape)
print("\nDataset information:")
print(df.info())

print("\nStatistical summary:")
print(df.describe())

# 1. Histograms of all columns
df.hist(figsize=(14, 10), bins=30)
plt.suptitle("Team 26 - Feature Distributions")
plt.tight_layout()
plt.savefig("histograms.png")
plt.show()

# 2. days_left against every input feature
features = [
    "battery_v",
    "temp_c",
    "rings_per_day",
    "wifi_rssi_dbm",
    "age_months"
]

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()

for i, feature in enumerate(features):
    axes[i].scatter(df[feature], df["days_left"], alpha=0.5)
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel("days_left")
    axes[i].set_title(feature + " vs days_left")

axes[5].axis("off")

plt.tight_layout()
plt.savefig("scatterplots.png")
plt.show()

print("\nPlots saved successfully!")