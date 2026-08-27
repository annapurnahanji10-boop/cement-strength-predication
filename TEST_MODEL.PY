import pandas as pd
import joblib

# Load trained model
model = joblib.load("models/concrete_model.pkl")

# Load scaler
scaler = joblib.load("models/concrete_scaler.pkl")

# Sample concrete input
sample = pd.DataFrame([[
    540.0,   # Cement
    0.0,     # Blast Furnace Slag
    0.0,     # Fly Ash
    162.0,   # Water
    2.5,     # Superplasticizer
    1040.0,  # Coarse Aggregate
    676.0,   # Fine Aggregate
    28       # Age
]], columns=[
    "Cement",
    "Blast Furnace Slag",
    "Fly Ash",
    "Water",
    "Superplasticizer",
    "Coarse Aggregate",
    "Fine Aggregate",
    "Age"
])

# Scale input
sample_scaled = scaler.transform(sample)

# Prediction
prediction = model.predict(sample_scaled)

print("=" * 50)
print("CONCRETE STRENGTH PREDICTION")
print("=" * 50)

print(f"Predicted Strength: {prediction[0]:.2f} MPa")