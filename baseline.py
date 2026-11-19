
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load cleaned dataset
df = pd.read_csv("data/team_26_clean.csv")

# Separate input features and target
X = df.drop("days_left", axis=1)
y = df["days_left"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Mean baseline
baseline = DummyRegressor(strategy="mean")
baseline.fit(X_train, y_train)
baseline_predictions = baseline.predict(X_test)

# Linear regression pipeline
linear_model = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    LinearRegression()
)

linear_model.fit(X_train, y_train)
linear_predictions = linear_model.predict(X_test)

# Evaluation function
def evaluate_model(name, actual, predicted):
    print(f"\n{name}")
    print("MAE:", mean_absolute_error(actual, predicted))
    print("RMSE:", mean_squared_error(actual, predicted) ** 0.5)
    print("R2 Score:", r2_score(actual, predicted))

# Compare models
evaluate_model("MEAN BASELINE", y_test, baseline_predictions)
evaluate_model("LINEAR REGRESSION", y_test, linear_predictions)

