
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# 1. Load original dataset
df = pd.read_csv("data/team_26_train.csv")

# 2. Mark invalid sensor readings as missing
df.loc[df["battery_v"] == 0, "battery_v"] = np.nan
df.loc[df["temp_c"] == -127, "temp_c"] = np.nan
df.loc[df["wifi_rssi_dbm"] == 0, "wifi_rssi_dbm"] = np.nan

# 3. Separate features and target
X = df.drop("days_left", axis=1)
y = df["days_left"]

# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# 5. Build the neural network pipeline
model = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    MLPRegressor(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        solver="adam",
        alpha=0.001,
        learning_rate_init=0.001,
        max_iter=1000,
        early_stopping=True,
        n_iter_no_change=30,
        random_state=42
    )
)

# 6. Train the neural network
print("\nTraining neural network...")
model.fit(X_train, y_train)

# 7. Make predictions
predictions = model.predict(X_test)

# 8. Evaluate the model
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\nNEURAL NETWORK RESULTS")
print("----------------------")
print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)

# 9. Count trainable parameters
mlp = model.named_steps["mlpregressor"]

parameters = sum(
    weights.size for weights in mlp.coefs_
) + sum(
    biases.size for biases in mlp.intercepts_
)

print("\nArchitecture: 5 -> 32 -> 16 -> 1")
print("Trainable parameters:", parameters)
print("Training iterations:", mlp.n_iter_)

# 10. Save trained model
joblib.dump(model, "battery_model.pkl")

print("\nModel saved as battery_model.pkl")
