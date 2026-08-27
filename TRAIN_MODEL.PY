import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from xgboost import XGBRegressor


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv(
    "dataset/concrete_data.csv",
    sep="\t",
    engine="python"
)

# Clean column names
df.columns = [
    "Cement",
    "Blast Furnace Slag",
    "Fly Ash",
    "Water",
    "Superplasticizer",
    "Coarse Aggregate",
    "Fine Aggregate",
    "Age",
    "Concrete Strength"
]


# ============================================================
# 2. DATA CLEANING
# ============================================================

# Convert to numeric
for column in df.columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# Remove missing values
df = df.dropna()

# Remove duplicates
df = df.drop_duplicates()

# Remove invalid values
input_columns = [
    "Cement",
    "Blast Furnace Slag",
    "Fly Ash",
    "Water",
    "Superplasticizer",
    "Coarse Aggregate",
    "Fine Aggregate",
    "Age"
]

for column in input_columns:
    df = df[df[column] >= 0]

df = df[df["Concrete Strength"] > 0]


# ============================================================
# 3. FEATURES AND TARGET
# ============================================================

X = df[input_columns]

y = df["Concrete Strength"]


# ============================================================
# 4. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# ============================================================
# 5. FEATURE SCALING
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


# ============================================================
# 6. CREATE MODELS
# ============================================================

models = {

    "Linear Regression":
        LinearRegression(),

    "Decision Tree":
        DecisionTreeRegressor(
            random_state=42
        ),

    "Random Forest":
        RandomForestRegressor(
            n_estimators=300,
            random_state=42
        ),

    "Gradient Boosting":
        GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        ),

    "XGBoost":
        XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            random_state=42,
            objective="reg:squarederror"
        )
}


# ============================================================
# 7. TRAIN AND EVALUATE MODELS
# ============================================================

results = []

trained_models = {}

print("\n" + "=" * 70)
print("MODEL TRAINING AND EVALUATION")
print("=" * 70)


for name, model in models.items():

    print(f"\nTraining {name}...")

    # Linear Regression and other models can use scaled data.
    # Tree-based models do not require scaling, but using the
    # same scaled features here keeps the comparison consistent.

    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    mse = mean_squared_error(
        y_test,
        y_pred
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_test,
        y_pred
    )

    results.append({
        "Model": name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2 Score": r2
    })

    trained_models[name] = model

    print(f"MAE      : {mae:.4f}")
    print(f"MSE      : {mse:.4f}")
    print(f"RMSE     : {rmse:.4f}")
    print(f"R2 Score : {r2:.4f}")


# ============================================================
# 8. MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="R2 Score",
    ascending=False
)

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# 9. SELECT BEST MODEL
# ============================================================

best_model_name = results_df.iloc[0]["Model"]

best_model = trained_models[
    best_model_name
]

print("\n" + "=" * 70)
print("BEST MODEL")
print("=" * 70)

print("Best Model:", best_model_name)
print(
    "Best R2 Score:",
    results_df.iloc[0]["R2 Score"]
)


# ============================================================
# 10. CREATE MODELS FOLDER
# ============================================================

import os

os.makedirs(
    "models",
    exist_ok=True
)


# ============================================================
# 11. SAVE BEST MODEL
# ============================================================

joblib.dump(
    best_model,
    "models/concrete_model.pkl"
)

joblib.dump(
    scaler,
    "models/concrete_scaler.pkl"
)


# ============================================================
# 12. SAVE MODEL RESULTS
# ============================================================

results_df.to_csv(
    "models/model_results.csv",
    index=False
)


print("\n" + "=" * 70)
print("MODEL SAVING")
print("=" * 70)

print("Best model saved:")
print("models/concrete_model.pkl")

print("Scaler saved:")
print("models/concrete_scaler.pkl")

print("Results saved:")
print("models/model_results.csv")

print("\nModel training completed successfully!")