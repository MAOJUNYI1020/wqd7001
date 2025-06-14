import pandas as pd
from joblib import load

# 1. Load the trained model
model = load("rf_model.pkl")

# 2. Create a sample input (must match training feature order)
sample = pd.DataFrame([{
    "temp": 29.5,
    "humidity": 78.0,
    "windspeed": 5.2,
    "aqi": 50,
    "precip": 1.8,
    "street_Jalan Sultan Salahuddin": 1,  # 1 = Jalan Sultan Salahuddin, 0 = Jalan Lapangan Terbang
    "hour": 8,
    "weekday": 2,
    "is_weekend": 0  # 0 = Weekday, 1 = Weekend
}])

# 3. Make prediction
prediction = model.predict(sample)[0]
proba = model.predict_proba(sample)

# 4. Display the result
print("Predicted Congestion Level:", prediction)
print("Prediction Probabilities (Smooth, Congested, Heavy):", proba)
